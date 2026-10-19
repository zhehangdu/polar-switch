# SPDX-License-Identifier: MIT
"""Search and verify Polar Switch circuits on two square buffers.

Each run finds the circuit of one outer iteration with k inner iterations,
verifies it by interval arithmetic, and rescales its buffers. There are
2**(k-1) buffer patterns. See coefficient_search/README.md for the commands,
options, and output files.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache, partial
import hashlib
import heapq
import importlib.metadata
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings

THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
# Each CLI worker uses one numerical-library thread. Set this before NumPy is
# loaded, so --workers is the only CPU-parallelism setting users need to choose.
if __name__ == '__main__':
    for thread_variable in THREAD_VARIABLES:
        os.environ[thread_variable] = '1'

import numpy as np  # noqa: E402

# SciPy is imported only inside the search functions, so verify needs only NumPy.

MODEL = 'alternating-two-buffer-v2'
SUMMARY_FORMAT = 'two-buffer-iteration-v2'
SAFETY = '1.005'                  # safety factor s of the rounding factors
LOWER = '0.001'                   # lower endpoint of the first input interval
DEFAULT_TOLERANCE = 1e-10         # requested verification accuracy
DEFAULT_MAX_BOXES = 500000        # branch-and-bound budget per extremum
PENALTY = 10                      # weight of the constraint violation in the merit
FEEDBACK_GAIN = 1e-7              # relative gain in r that continues the feedback cycles
SQP_MAXITER = 30                  # SLSQP iterations per local step
ROUNDING_MARGIN = Fraction(999999999999, 1000000000000)  # retry factor after rounding
STORE_MIN, STORE_MAX = 1/16, 1024  # range of every hidden store after export
FP16_MAX = 65504.
STATE_CEILING = FP16_MAX/2        # largest stored value allowed after export
RANGE_BINS = 16384                # input bins of the exported store ranges
ROOT = Path(__file__).resolve().parent


# Circuits and their evaluation.

@dataclass(eq=False)
class Circuit:
    """Coefficients of one outer iteration on two buffers, A and B.

    Instruction j, (w, p, q, t) with rows[j] = (a, b, c), computes
    buffer[w] = a*buffer[p]*buffer[q] + b*buffer[t] + c*I; these are the
    paper's (w, i, j, r) and (a, b, c). Buffer 0 (A) starts with the Gram
    matrix, p = q is the buffer written last, and the destinations alternate.
    On a singular value x, the circuit returns x*h(x**2), where h is the last
    buffer written. The last row includes the output scaling.
    """
    instructions: tuple
    rows: np.ndarray

    def __post_init__(self):
        """Check the alternating buffer pattern and the coefficient rows."""
        self.instructions = tuple(tuple(int(i) for i in step) for step in self.instructions)
        self.rows = np.array(self.rows, dtype=float)
        for j, (w, p, q, t) in enumerate(self.instructions):
            if (w, p, q) != (1-j % 2, j % 2, j % 2) or t not in ((0,) if j == 0 else (0, 1)):
                raise ValueError('Each instruction must square the latest buffer and alternate destinations')
        if (not self.instructions or self.rows.shape != (len(self.instructions), 3)
                or not np.isfinite(self.rows).all()):
            raise ValueError('A circuit needs one finite (a, b, c) row per instruction')

    @property
    def depth(self):
        """Return the number of inner iterations k."""
        return len(self.instructions)

    @property
    def dim(self):
        """Return the number of variables: the input x and the rounding factors v_0, ..., v_k."""
        return self.depth + 2

    def vector(self):
        """Return the coefficient rows as one parameter vector."""
        return self.rows.flatten()

    def with_vector(self, vector):
        """Return a circuit with the coefficient rows of a parameter vector."""
        return Circuit(self.instructions, np.reshape(vector, (self.depth, 3)))

    def record(self):
        """Return a JSON record with exact hexadecimal coefficients."""
        return {'model': MODEL,
                'instructions': [list(step) for step in self.instructions],
                'rows': self.rows.tolist(),
                'rows_hex': [[float(v).hex() for v in row] for row in self.rows]}

    @classmethod
    def from_record(cls, record):
        """Read a circuit from a record written by record()."""
        if record.get('model') != MODEL:
            raise ValueError('Unsupported circuit model')
        return cls(record['instructions'], [[float.fromhex(v) for v in row] for row in record['rows_hex']])


def scale_output(circuit, factor):
    """Multiply the last row by a positive rational factor, rounding to binary64."""
    factor = Fraction(factor)
    rows = circuit.rows.copy()
    rows[-1] = [float(Fraction(float(value))*factor) for value in rows[-1]]
    return Circuit(circuit.instructions, rows)


def domain(circuit, safety, lower=Fraction(1, 1000), inward=False, upper=1):
    """Return the lower and upper corners of the variable box.

    The input ranges over [lower, upper], and each rounding factor over
    [1/safety, safety]. Endpoints are rounded outward, or inward if inward is True.
    """
    s, ell, upper = Fraction(safety), Fraction(lower), Fraction(upper)
    if s < 1 or not 0 <= ell <= upper or upper <= 0:
        raise ValueError("Require safety >= 1 and 0 <= lower <= upper, upper > 0")
    vlo = rational_bounds(1/s)[int(inward)]
    vhi = rational_bounds(s)[int(not inward)]
    return (np.r_[rational_bounds(ell)[int(inward)], np.full(circuit.dim-1, vlo)],
            np.r_[rational_bounds(upper)[int(not inward)], np.full(circuit.dim-1, vhi)])


def rational_bounds(value):
    """Return the binary64 values just below and above an exact rational, with overflow to infinity."""
    exact = Fraction(value)
    try:
        nearest = float(exact)
    except OverflowError:
        largest = sys.float_info.max
        return (largest, math.inf) if exact > 0 else (-math.inf, -largest)
    frozen = Fraction(nearest)
    return (math.nextafter(nearest, -math.inf) if frozen > exact else nearest,
            math.nextafter(nearest, math.inf) if frozen < exact else nearest)


def patterns(depth):
    """Yield the label and instructions of every alternating buffer pattern with depth steps.

    The first inner iteration adds the Gram matrix. In each later iteration,
    bit 0 adds the buffer written last and bit 1 the older buffer.
    """
    for bits in product((0, 1), repeat=depth-1):
        instructions = [(1, 0, 0, 0)]
        for j, bit in enumerate(bits, 1):
            p = j % 2
            instructions.append((1-p, p, p, p if bit == 0 else 1-p))
        yield ''.join(map(str, bits)), tuple(instructions)


def evaluate(circuit, z, parameters=False, variables=False, quotient=False, both=False):
    """Evaluate x*h(x**2) without the rounding factor v_{k+1} of the output.

    Each row of z holds the input x and the rounding factors v_0, ..., v_k.
    parameters returns derivatives for the flattened coefficient rows, and
    variables returns derivatives for the columns of z. quotient returns h
    instead, so that the sign of h is also checked at x = 0. both returns the
    output and h, each with its derivatives, from one evaluation.
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    dim =3*circuit.depth if parameters else circuit.dim if variables else 0
    n = len(z)
    x, v0 = z[:, 0], z[:, 1]
    if not dim:
        buffers = [v0*x*x, None]
        for j, ((w, p, q, t), (a, b, c)) in enumerate(zip(circuit.instructions, circuit.rows)):
            buffers[w] = z[:, j+2]*(a*(buffers[p]*buffers[q])+b*buffers[t]+c)
        factor = buffers[circuit.instructions[-1][0]]
        if quotient:
            return factor
        return (x*factor, factor) if both else x*factor
    dx = np.zeros((n, dim))
    if variables:
        dx[:, 0] = 1.
    gram = v0*x*x
    dgram = 2*(v0*x)[:, None]*dx
    if variables:
        dgram[:, 1] += x*x
    buffers = [gram, None]
    derivatives = [dgram, None]
    for j, ((w, p, q, t), (a, b, c)) in enumerate(zip(circuit.instructions, circuit.rows)):
        # Snapshot every operand before assignment, including an aliased addend.
        left, right, addend = buffers[p], buffers[q], buffers[t]
        dleft, dright, daddend = derivatives[p], derivatives[q], derivatives[t]
        v = z[:, j+2]
        product = left*right
        inner = a*product+b*addend+c
        current = v*inner
        dcurrent = v[:, None]*(a*(right[:, None]*dleft+left[:, None]*dright)
                              + b*daddend)
        if parameters:
            dcurrent[:, 3*j] += v*product
            dcurrent[:, 3*j+1] += v*addend
            dcurrent[:, 3*j+2] += v
        if variables:
            dcurrent[:, j+2] += inner
        buffers[w], derivatives[w] = current, dcurrent
    output = circuit.instructions[-1][0]
    factor, factor_derivative = buffers[output], derivatives[output]
    if quotient:
        return factor, factor_derivative
    derivative = factor[:, None]*dx+x[:, None]*factor_derivative
    if both:
        return x*factor, derivative, factor, factor_derivative
    return x*factor, derivative


def last_row_features(circuit, z):
    """Return the derivatives of the output and of h with respect to the last row at z.

    The output is linear in the last row (a, b, c): h = v_k*(a*B_p**2 + b*B_t + c),
    so only the values of the earlier stores are needed.
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    x = z[:, 0]
    buffers = [z[:, 1]*x*x, None]
    for j, ((w, p, q, t), (a, b, c)) in enumerate(zip(circuit.instructions[:-1], circuit.rows[:-1])):
        buffers[w] = z[:, j+2]*(a*(buffers[p]*buffers[q])+b*buffers[t]+c)
    _, p, q, t = circuit.instructions[-1]
    v = z[:, circuit.depth+1]
    factor = np.c_[v*(buffers[p]*buffers[q]), v*buffers[t], v]
    return x[:, None]*factor, factor


# Outward binary64 arithmetic. Every elementary result encloses exact arithmetic.

NEG_INF, POS_INF = np.float64(-np.inf), np.float64(np.inf)


def down(v):
    """Return the next binary64 value below v."""
    return np.nextafter(v, NEG_INF)


def up(v):
    """Return the next binary64 value above v."""
    return np.nextafter(v, POS_INF)


class I:
    """Intervals of binary64 values with outward rounding.

    Every operation returns intervals that contain the exact results for all
    values in its operands.
    """
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(lo if hi is None else hi, dtype=float)

    @classmethod
    def _wrap(cls, lo, hi):
        """Return an interval with float endpoints computed by NumPy, without conversion."""
        value = object.__new__(cls)
        value.lo, value.hi = lo, hi
        return value

    def __add__(self, other):
        if isinstance(other, I):
            return I._wrap(down(self.lo+other.lo), up(self.hi+other.hi))
        return I._wrap(down(self.lo+other), up(self.hi+other))

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, I):
            ll, lh = self.lo*other.lo, self.lo*other.hi
            hl, hh = self.hi*other.lo, self.hi*other.hi
            return I._wrap(down(np.minimum(np.minimum(ll, lh), np.minimum(hl, hh))),
                           up(np.maximum(np.maximum(ll, lh), np.maximum(hl, hh))))
        # A number has equal endpoints, so two of the four products suffice.
        lo, hi = self.lo*other, self.hi*other
        return I._wrap(down(np.minimum(lo, hi)), up(np.maximum(lo, hi)))

    __rmul__ = __mul__

    def square(self):
        """Return the range of the square."""
        l2, h2 = self.lo*self.lo, self.hi*self.hi
        return I._wrap(np.where((self.lo <= 0) & (self.hi >= 0), 0., down(np.minimum(l2, h2))),
                       up(np.maximum(l2, h2)))

    def sum(self):
        """Sum along the last axis, rounding each addition outward."""
        # The error of a NumPy sum cannot be bounded by one nextafter.
        result = I(np.zeros(self.lo.shape[:-1]))
        for j in range(self.lo.shape[-1]):
            result = result + I(self.lo[..., j], self.hi[..., j])
        return result


# Interval automatic differentiation, including shared quadratic reads.

@lru_cache(maxsize=1024)
def _vertex(a, b, c):
    """Enclose the location and value of the vertex of a*x**2 + b*x + c."""
    a, b, c = map(Fraction.from_float, (a,b,c))
    return (*rational_bounds(-b/(2*a)), *rational_bounds(c-b*b/(4*a)))


def quadratic_range(value, a, b, c):
    """Enclose the complete range of a*X**2+b*X+c on an interval X.

    Scalar coefficients denote their exact finite binary64 values. An outward
    vertex-location interval overlapping X is included conservatively; a
    rounded location can therefore never cause an interior extremum to vanish.
    """
    if a == 0:
        return b*value+c if b != 0 else I(np.full_like(value.lo,c))
    endpoints = I(np.stack((value.lo,value.hi)))
    endpoint_values = a*endpoints.square()+b*endpoints+c
    lower = np.min(endpoint_values.lo,axis=0)
    upper = np.max(endpoint_values.hi,axis=0)
    vertex_lo,vertex_hi,minimum,maximum = _vertex(float(a),float(b),float(c))
    overlaps = (value.lo <= vertex_hi) & (vertex_lo <= value.hi)
    if a > 0:
        lower = np.where(overlaps,np.minimum(lower,minimum),lower)
    else:
        upper = np.where(overlaps,np.maximum(upper,maximum),upper)
    return I(lower,upper)


def graph(circuit, lo, hi, derivatives=True, quotient=False):
    """Enclose the output and its gradient over boxes of variables.

    lo and hi hold the corners of each box. An inner iteration whose square and
    addend read the same buffer is enclosed as one quadratic, which gives
    tighter ranges. quotient omits the outer x, as in evaluate.
    """
    lo,hi = np.atleast_2d(lo),np.atleast_2d(hi)
    n,dim = lo.shape
    basis = np.eye(dim)

    def variable(column):
        value = I(lo[:,column:column+1],hi[:,column:column+1])
        gradient = I(np.broadcast_to(basis[column],(n,dim))) if derivatives else None
        return value,gradient

    x,dx = variable(0)
    v0,dv0 = variable(1)
    buffers = [v0*x.square(),None]
    gradients = [dv0*x.square()+2*v0*x*dx,None] if derivatives else None
    for j,((w,p,q,t),(a,b,c)) in enumerate(zip(circuit.instructions,circuit.rows)):
        # Read all original sources before a destination/addend overwrite.
        bp,bq,bt = buffers[p],buffers[q],buffers[t]
        same = p == t
        if same:
            inner = quadratic_range(bp,a,b,c)
        else:
            product = bp.square() if p == q else bp*bq
            inner = a*product+b*bt+c
        v,dv = variable(j+2)
        if derivatives:
            dp,dq,dt = gradients[p],gradients[q],gradients[t]
            if same:
                dinner = (a*(2*bp)+b)*dp
            else:
                dproduct = 2*bp*dp if p == q else dp*bq+bp*dq
                dinner = a*dproduct+b*dt
            current_gradient = dv*inner+v*dinner
        current = v*inner
        buffers[w] = current
        if derivatives:
            gradients[w] = current_gradient
    value = buffers[circuit.instructions[-1][0]]
    gradient = gradients[circuit.instructions[-1][0]] if derivatives else None
    if not quotient:
        if derivatives:
            gradient = dx*value+x*gradient
        value = x*value
    return value,gradient


# Exact rational bounds, used to show that the final factor h is nonnegative.

def _multiply(left, right):
    """Return the exact range of a product of two intervals."""
    values = [x*y for x in left for y in right]
    return min(values), max(values)


def _scale(value, factor):
    """Return the exact range of an interval multiplied by a number."""
    values = [factor*endpoint for endpoint in value]
    return min(values), max(values)


def _square(value):
    """Return the exact range of the square of an interval."""
    lower, upper = value
    endpoints = [lower*lower, upper*upper]
    return (Fraction(0) if lower <= 0 <= upper else min(endpoints), max(endpoints))


def _quadratic(value, a, b, c):
    """Return the exact range of a*x**2 + b*x + c on an interval."""
    lower, upper = value
    points = [lower, upper]
    if a:
        vertex = -b/(2*a)
        if lower <= vertex <= upper:
            points.append(vertex)
    values = [a*x*x+b*x+c for x in points]
    return min(values), max(values)


def _row_range(buffers, p, t, a, b, c):
    """Return the exact range of one inner iteration before its store."""
    if p == t:
        return _quadratic(buffers[p], a, b, c), 'shared_quadratic'
    product = _scale(_square(buffers[p]), a)
    addend = _scale(buffers[t], b)
    return (product[0]+addend[0]+c, product[1]+addend[1]+c), 'square_plus_addend'


def _exact_record(value):
    """Return an exact rational as hexadecimal numerator and denominator."""
    # Hex avoids Python's decimal integer-string limit for unusually large
    # exact intermediates, while retaining both rational endpoints exactly.
    return {'numerator_hex': hex(value.numerator), 'denominator_hex': hex(value.denominator)}


def _interval_record(value):
    """Return the exact and outward-rounded endpoints of an interval."""
    lower, upper = rational_bounds(value[0])[0], rational_bounds(value[1])[1]
    return {'exact_interval': [_exact_record(endpoint) for endpoint in value],
            'outward_interval': [endpoint if math.isfinite(endpoint) else None
                                 for endpoint in (lower, upper)],
            'outward_hex': [lower.hex(), upper.hex()]}


def factor_bounds(circuit, safety, upper):
    """Bound every stored value, and the final factor h, in exact rational arithmetic.

    Propagates intervals for inputs in [0, upper] and rounding factors in
    [1/safety, safety]. Returns float_lower, an outward lower bound on h;
    nonnegative, which is True when the exact bounds prove h >= 0; and a trace
    of the sources, coefficients, and exact range of every store. These bounds
    are too loose to choose the output scaling.
    """
    s, input_upper = Fraction(safety), Fraction(upper)
    multiplier = (1/s, s)
    buffers = [(Fraction(0), s*input_upper**2), None]
    versions = [0, None]
    trace = [{'version': 0, 'slot': 0, 'method': 'initial_gram',
              **_interval_record(buffers[0])}]
    for version, ((w, p, q, t), row) in enumerate(zip(circuit.instructions, circuit.rows), 1):
        a, b, c = (Fraction.from_float(float(value)) for value in row)
        inner, method = _row_range(buffers, p, t, a, b, c)
        current = _multiply(inner, multiplier)
        trace.append({'version': version, 'slot': w, 'sources': [versions[p], versions[q], versions[t]],
                      'method': method, 'rows_hex': [float(value).hex() for value in row],
                      'pre_store': _interval_record(inner), **_interval_record(current)})
        buffers[w], versions[w] = current, version
    lower = buffers[circuit.instructions[-1][0]][0]
    return {'float_lower': rational_bounds(lower)[0], 'nonnegative': lower >= 0, 'trace': trace}


# Extrema over the whole input interval by interval subdivision.

def bound(c, lo, hi, sign, graph_function):
    """Bound sign*output from below on each box with a mean value form.

    Coordinates with a gradient of fixed sign are first fixed at the minimizing
    face. Returns the boxes, the lower bounds, the box midpoints, and the split
    score of each coordinate.
    """
    lo, hi = lo.copy(), hi.copy()
    for _ in range(2):
        _, grad = graph_function(c, lo, hi)
        grad = grad*sign
        inc, dec = grad.lo > 0, grad.hi < 0
        hi[inc] = lo[inc]
        lo[dec] = hi[dec]
    value, grad = graph_function(c, lo, hi)
    value, grad = sign*value, sign*grad
    mid = lo+(hi-lo)*.5
    point, point_grad = graph_function(c, mid, mid)
    correction = (grad*I(down(lo-mid), up(hi-mid))).sum()
    mean = sign*point+I(correction.lo[:, None], correction.hi[:, None])
    lower = np.maximum(value.lo[:, 0], mean.lo[:, 0])
    scores = (hi-lo)*np.maximum(abs(grad.lo), abs(grad.hi))
    fallback = (hi-lo)*np.maximum(abs(point_grad.lo), abs(point_grad.hi))
    fallback = np.where(np.isfinite(fallback), fallback, hi-lo)
    fallback += (hi-lo)*1e-100
    scores = np.where(np.isfinite(scores), scores, fallback)
    scores = np.where(hi > lo, np.maximum(scores, (hi-lo)*1e-100), 0.)
    bad = ~np.isfinite(lower)
    # An overflowing interval bound does not mean the circuit overflows. Keep the
    # box with lower bound -infinity and split it along the coordinate with the
    # largest point sensitivity.
    if bad.any():
        lower[bad] = -np.inf
        scores[bad] = fallback[bad]
    return lo, hi, lower, mid, scores

def extremum(c, sign, tolerance, max_boxes, graph_function, corners, lower_target=None):
    """Bound the minimum of sign*output over a box of variables by branch and bound.

    corners = (dl, dh, il, ih) holds the outward corners of the box and the
    inward corners used for point evaluations. Boxes are split along the
    coordinate with the largest score until the lower bound is within tolerance
    of the smallest value found at a point, until it reaches lower_target, or
    until max_boxes boxes have been processed. Returns the status, the
    enclosure [lower bound, best point value], the best point, and the work done.
    """
    start = time.monotonic()
    dl, dh, il, ih = corners
    # A fixed input value needs one box, not 128 identical ones.
    edges = np.array([dl[0], dh[0]]) if dl[0] == dh[0] else np.linspace(dl[0], dh[0], 129)
    if dl[0] > 0 and dh[0] > dl[0]:
        edges = np.unique(np.r_[edges, np.geomspace(dl[0], dh[0], 65)])
    lo, hi = np.tile(dl, (len(edges)-1, 1)), np.tile(dh, (len(edges)-1, 1))
    lo[:, 0], hi[:, 0] = edges[:-1], edges[1:]
    heap = []
    uid = 0
    witness = np.inf
    wz = None
    retired = np.inf
    processed = 0
    stalled_boxes = 0
    history = []
    status = 'unfinished'
    while True:
        lo, hi, lb, mid, scores = bound(c, lo, hi, sign, graph_function)
        processed += len(lo)
        points = np.clip(mid, il, ih)
        pv, _ = graph_function(c, points, points, derivatives=False)
        pv = pv*sign
        if not np.isfinite(pv.hi).all():
            return {'status': 'nonfinite_point_enclosure', 'boxes': processed}
        ix = int(np.argmin(pv.hi[:, 0]))
        if pv.hi[ix, 0] < witness:
            witness = float(pv.hi[ix, 0])
            wz = points[ix].tolist()
        for j in range(len(lo)):
            if lb[j] >= witness-tolerance:
                retired = min(retired, float(lb[j]))
            else:
                heapq.heappush(heap, (float(lb[j]), uid, lo[j], hi[j], scores[j]))
                uid += 1
        lower = min(retired, heap[0][0] if heap else witness)
        if len(history) < 300 or processed > 2*history[-1][0]:
            history.append([processed, len(heap), lower, witness])
        if lower_target is not None and lower >= lower_target:
            status = 'lower_target_certified'
            break
        if witness-lower <= tolerance:
            status = 'certified'
            break
        if processed >= max_boxes:
            break
        batch = []
        while heap and len(batch) < 128:
            item = heapq.heappop(heap)
            if item[0] >= witness-tolerance:
                retired = min(retired, item[0])
            else:
                batch.append(item)
        if not batch:
            lower = min(retired, witness)
            status = 'certified' if witness-lower <= tolerance else 'roundoff_stall'
            break
        lows, highs = [], []
        for item_lower, _, l, h, sc in batch:
            axis = int(np.argmax(sc))
            middle = l[axis]+(h[axis]-l[axis])*.5
            if middle == l[axis] or middle == h[axis]:
                # Keep its conservative bound, but do not let one point-sized
                # rounding floor stop refinement of all the other boxes.
                retired = min(retired, item_lower)
                stalled_boxes += 1
                continue
            lh, rl = h.copy(), l.copy()
            lh[axis] = rl[axis] = middle
            lows.extend((l, rl))
            highs.extend((lh, h))
        if not lows:
            if not heap:
                lower = min(retired, witness)
                status = 'certified' if witness-lower <= tolerance else 'roundoff_stall'
                break
            # Reprocess another box; previously stalled leaves stay retired.
            item = heapq.heappop(heap)
            lows.append(item[2])
            highs.append(item[3])
        lo, hi = np.array(lows), np.array(highs)
    if not np.isfinite(lower):
        return {'status': 'unresolved_nonfinite_boxes', 'witness_bound': witness,
                'witness': wz, 'boxes': processed, 'seconds': time.monotonic()-start}
    history = [[boxes, queued, low if np.isfinite(low) else None, best]
               for boxes, queued, low, best in history]
    return {'status': status, 'enclosure': [lower, witness], 'witness': wz,
            'boxes': processed, 'stalled_boxes': stalled_boxes,
            'seconds': time.monotonic()-start, 'history': history}


# Verification of one circuit.

def certify(circuit, safety, lower, tolerance, max_boxes, upper=1, output_upper=None):
    """Verify a circuit over its whole input interval.

    Three extrema are bounded by branch and bound: the minimum of h over inputs
    in [0, upper] and all rounding factors, which must be nonnegative (exact
    rational bounds may prove this first); the maximum of the output over the
    same domain, multiplied by safety for the output rounding factor; and the
    minimum of the exact output over [lower, upper], which is the verified
    lower endpoint r. ratio is a lower bound on r divided by that maximum; it
    does not depend on the output scale, so the search compares circuits
    without rescaling them. If output_upper is given, the maximum must not
    exceed it.

    max_boxes applies to each extremum. feasible is False when the checks did
    not succeed; this does not prove that the circuit is infeasible.
    """
    started = time.monotonic()
    s, ell, upper = map(Fraction, (safety, lower, upper))
    cap_output = None if output_upper is None else Fraction(output_upper)
    sl, sh = rational_bounds(s)
    parts = {}
    out = {'status': 'unresolved', 'feasible': False, 'r': None, 'ratio': None,
           'circuit': circuit.record(),
           'safety': str(s), 'lower_endpoint_domain': [str(ell), str(upper)],
           'output_cap': None if cap_output is None else str(cap_output), 'parts': parts,
           'requested_accuracy': tolerance, 'max_boxes': max_boxes}

    def finish(reason=None):
        out['seconds'] = time.monotonic()-started
        out['boxes_processed'] = sum(part.get('boxes', 0) for part in parts.values())
        if reason is not None:
            out['unresolved_reason'] = reason
        return out

    def record_part(name, evidence):
        parts[name] = evidence
        bracket = evidence.get('enclosure')
        if bracket is None or not np.isfinite(bracket).all() or bracket[0] > bracket[1]:
            return None
        return bracket

    def run(name, sign, input_lower, tol, quotient=False, nominal=False, lower_target=None):
        part_safety = Fraction(1) if nominal else s
        corners = (*domain(circuit, part_safety, lower=input_lower, upper=upper),
                   *domain(circuit, part_safety, lower=input_lower, inward=True, upper=upper))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            evidence = extremum(circuit, sign, tol, max_boxes, partial(graph, quotient=quotient),
                                corners, lower_target=lower_target)
        return record_part(name, evidence)

    # Two probe points set the scale of the requested accuracy.
    probe = np.ones((2, circuit.dim))
    probe[:, 0] = [float(upper)/2, rational_bounds(upper)[0]]

    def exact_factor_proof():
        """Try to prove h >= 0 with exact rational bounds before branch and bound."""
        proof_started = time.monotonic()
        states = factor_bounds(circuit, s, upper)
        if not states['nonnegative']:
            return None
        # The exact bounds give a lower bound on h; a probe point gives an upper
        # bound on its minimum.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            values, _ = graph(circuit, probe, probe, derivatives=False, quotient=True)
        finite = (np.isfinite(values.lo[:, 0]) & np.isfinite(values.hi[:, 0])
                  & (values.lo[:, 0] <= values.hi[:, 0])
                  & (values.hi[:, 0] >= states['float_lower']))
        if not finite.any():
            return None
        index = int(np.argmin(np.where(finite, values.hi[:, 0], np.inf)))
        evidence = {
            'status': 'lower_target_certified',
            'method': 'exact-rational-state-rectangle-positivity',
            'enclosure': [states['float_lower'], float(values.hi[index, 0])],
            'witness': probe[index].tolist(), 'boxes': 0, 'stalled_boxes': 0,
            'history': [],
            'seconds': time.monotonic()-proof_started,
            'exact_lower': deepcopy(states['trace'][-1]['exact_interval'][0]),
            'state_trace': deepcopy(states['trace']),
            'safety': str(s), 'input_domain': ['0', str(upper)],
            'scope': 'h >= 0 only; the outer x is excluded',
        }
        return record_part('factor_min', evidence)

    with np.errstate(over='ignore', invalid='ignore'):
        values, _ = graph(circuit, probe, probe, derivatives=False)
    magnitudes = np.abs(np.r_[values.lo.ravel(), values.hi.ravel()])
    finite = magnitudes[np.isfinite(magnitudes)]
    scale = max(float(finite.max()) if finite.size else 1., 1e-100)
    work_tolerance = max(tolerance*.02*scale, np.nextafter(0., 1.))

    factor = exact_factor_proof()
    if factor is None:
        factor = run('factor_min', 1, Fraction(0), work_tolerance,
                     quotient=True, lower_target=0.)
    maximum = run('full_negative_max', -1, Fraction(0), work_tolerance)
    minimum = run('performance_min', 1, ell, work_tolerance, nominal=True)
    if any(value is None for value in (factor, maximum, minimum)):
        return finish('One or more continuous extrema lack finite enclosures')
    if factor[0] < 0 or minimum[0] <= 0 or -maximum[1] <= 0:
        if factor[1] < 0 or minimum[1] <= 0:
            out['status'] = 'infeasible'
        return finish('Could not verify h >= 0 or a positive lower endpoint')

    # Because h >= 0, the output is largest when the output rounding factor
    # equals safety. The lower endpoint uses unit rounding factors.
    nominal = I(*minimum)
    robust = I(-maximum[1], -maximum[0])*I(sl, sh)
    if not np.isfinite([robust.lo, robust.hi]).all() or robust.lo <= 0:
        return finish('Output maximum enclosure is nonfinite or nonpositive')
    gap = float(up(max(nominal.hi-nominal.lo, robust.hi-robust.lo)))
    feasible = bool(nominal.lo > 0 and (cap_output is None or Fraction(float(robust.hi)) <= cap_output))
    ratio = rational_bounds(Fraction(float(nominal.lo))/Fraction(float(robust.hi)))[0]
    out.update(r=float(nominal.lo), r_hex=float(nominal.lo).hex(), ratio=ratio,
               nominal_minimum_bracket=[float(nominal.lo), float(nominal.hi)],
               robust_maximum_bracket=[float(robust.lo), float(robust.hi)],
               gap=gap, feasible=feasible, accuracy_reached=bool(gap <= tolerance),
               status='verified' if feasible else 'unresolved')
    return finish()


# Power-of-two buffer scaling chosen from sample points.

def _samples(circuit, seed, safety):
    """Return nominal, random interior, and random corner sample points."""
    x = np.unique(np.r_[np.geomspace(float(LOWER), 1., 49), np.linspace(0., 1., 49)])
    nominal = np.ones((len(x), circuit.dim))
    nominal[:, 0] = x
    lo, hi = domain(circuit, safety, lower=0., inward=True)
    rng = np.random.default_rng(seed)
    interior = nominal.copy()
    interior[:, 1:] = lo[1:]+(hi[1:]-lo[1:])*rng.random((len(x), circuit.dim-1))
    corners = nominal.copy()
    corners[:, 1:] = np.where(rng.integers(0, 2, (len(x), circuit.dim-1)),
                              hi[1:], lo[1:])
    return np.r_[nominal, interior, corners]


def _trace(circuit, z):
    """Return the largest absolute value of every store and the output at z."""
    gram = z[:, 1]*z[:, 0]*z[:, 0]
    buffers = [gram, None]
    maxima = [float(np.max(abs(gram)))]
    if not np.isfinite(gram).all():
        return None
    for j, ((w, p, q, t), (a, b, c)) in enumerate(zip(circuit.instructions, circuit.rows)):
        # Read all sources before updating an aliased destination addend.
        current = z[:, j+2]*(a*(buffers[p]*buffers[q])+b*buffers[t]+c)
        if not np.isfinite(current).all():
            return None
        maxima.append(float(np.max(abs(current))))
        buffers[w] = current
    output = z[:, 0]*buffers[circuit.instructions[-1][0]]
    if not np.isfinite(output).all():
        return None
    return maxima, output


def _normalizing_exponent(magnitude):
    """Return the exponent of the power of two that brings magnitude closest to one."""
    if not magnitude:
        return 0
    fraction, exponent = math.frexp(magnitude)
    # Choose the nearest power of two: normalized maxima lie near [sqrt(.5),sqrt(2)].
    return (1 if fraction <= math.sqrt(.5) else 0)-exponent


def _shift_exact(value, exponent):
    """Return value*2**exponent, or None if the result is not exact."""
    value = float(value)
    if value == 0:
        return value
    try:
        scaled = math.ldexp(value, exponent)
        if not math.isfinite(scaled) or scaled == 0:
            return None
        recovered = math.ldexp(scaled, -exponent)
    except OverflowError:
        return None
    # In particular, reject rounded subnormal shifts even when nonzero survives.
    return scaled if recovered.hex() == value.hex() else None


def normalize_states(circuit, *, seed, safety):
    """Rescale the hidden stores by powers of two to bring their values near one.

    The scale of each store is chosen from its largest value at nominal,
    interior, and corner sample points over [0, 1]; the samples choose the
    scales only. The coefficients are adjusted so that the output does not
    change, and the Gram store and the last store keep scale one. Every shift
    must be exact in binary64; if a shift would overflow, underflow, or round,
    or a sample is not finite, the original circuit is returned with
    applied=False.
    """
    points = _samples(circuit, seed, safety)
    info = {'applied': False, 'exact': False, 'status': 'unfinished',
            'sample_count': len(points), 'sample_safety': str(safety), 'seed': int(seed),
            'state_scale_exponents': [0]*(circuit.depth+1),
            'requested_state_scale_exponents': None,
            'state_max_before': None, 'state_max_after': None,
            'sampled_relative_output_error': None}

    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        original = _trace(circuit, points)
        if original is None:
            info['status'] = 'nonfinite_original_trajectory'
            return circuit, info
        maxima, original_output = original
        exponents = [0]+[_normalizing_exponent(value) for value in maxima[1:-1]]+[0]
        info.update(state_max_before=maxima, state_max_after=list(maxima),
                    requested_state_scale_exponents=exponents)
        versions = [0, None]
        rows = np.empty_like(circuit.rows)
        shifts = []
        for j, (w, p, q, t) in enumerate(circuit.instructions, 1):
            new = exponents[j]
            row_shifts = [new-exponents[versions[p]]-exponents[versions[q]],
                          new-exponents[versions[t]], new]
            shifts.append(row_shifts)
            for column, exponent in enumerate(row_shifts):
                value = _shift_exact(circuit.rows[j-1, column], exponent)
                if value is None:
                    info.update(status='unsafe_coefficient_shift',
                                rejected_coefficient=[j-1, column],
                                rejected_shift=exponent)
                    return circuit, info
                rows[j-1, column] = value
            versions[w] = j
        transformed = Circuit(circuit.instructions, rows)
        replay = _trace(transformed, points)
        if replay is None:
            info['status'] = 'nonfinite_transformed_trajectory'
            return circuit, info
        new_maxima, output = replay
        output_scale = max(1., float(np.max(abs(original_output))))
        discrepancy = float(np.max(abs(output/output_scale-original_output/output_scale)))
        applied = any(exponents)
        info.update(applied=applied, exact=True,
                    status='scaled' if applied else 'already_balanced',
                    state_scale_exponents=exponents, state_max_after=new_maxima,
                    row_coefficient_shifts=shifts,
                    sampled_relative_output_error=discrepancy)
        return transformed, info


# Optimization on sample points. Candidates are verified separately.

def solve_lp(*args, **kwargs):
    """Solve a linear program with HiGHS on one thread with fixed tolerances."""
    from scipy.optimize import linprog, OptimizeWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        return linprog(*args, **kwargs, method='highs', options={'threads': 1,
            'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})


def nominal_points(c, z, lower):
    """Return the sample inputs in [lower, 1] with every rounding factor equal to one."""
    x = np.unique(np.r_[float(lower), z[z[:, 0] >= lower, 0], 1.])
    points = np.ones((len(x), c.dim))
    points[:, 0] = x
    return points


def guards(c, safety, lower, seed):
    """Return the sample points used as constraints during optimization.

    They combine 1024 Sobol points over the inputs and rounding factors, a grid
    of inputs with unit rounding factors, random corners, and for each grid input
    the rounding factors that most increase the output or decrease h to first order.
    """
    from scipy.stats import qmc
    lo, hi = domain(c, safety, lower=0., inward=True)
    u = qmc.Sobol(c.dim, scramble=True, seed=seed).random_base2(10)
    u[len(u)//2:, 1:] = u[len(u)//2:, 1:] >= .5
    z = lo+(hi-lo)*u
    z[::2, 0] = np.exp(np.log(lower)*(1-u[::2, 0]))
    x = np.unique(np.r_[0., np.linspace(0, lower, 33),
        np.geomspace(lower, 1, 161), np.linspace(lower, 1, 161)])
    nominal = np.ones((len(x), c.dim)); nominal[:, 0] = x
    if Fraction(safety) == 1:
        return nominal
    _, jac, _, factor_jac = evaluate(c, nominal, variables=True, both=True)
    plus, minus = nominal.copy(), nominal.copy()
    plus[:, 1:] = np.where(jac[:, 1:] >= 0, hi[1:], lo[1:])
    minus[:, 1:] = np.where(factor_jac[:, 1:] >= 0, lo[1:], hi[1:])
    rng = np.random.default_rng(seed)
    extra = lo+(hi-lo)*(rng.random((128, c.dim)) >= .5)
    extra[:64, 0] = lower
    return np.unique(np.r_[z, nominal, extra, plus, minus], axis=0)


def adversaries(c, safety, lower, guard, starts=6, guard_nominal=None):
    """Find local worst cases with L-BFGS-B, starting from the worst samples.

    Searches for the smallest h, the largest output, and the smallest output with
    unit rounding factors in [lower, 1], from up to starts distinct points each.
    guard_nominal holds nominal_points of guard, if already computed.
    """
    from scipy.optimize import minimize
    points = []
    if guard_nominal is None:
        guard_nominal = nominal_points(c, guard, lower)
    # Both safety adversaries retain all stores; the performance adversary has
    # only one free coordinate, its input in [lower,1].
    for sign, low, nominal, quotient in ((1, 0., False, True),
            (-1, 0., False, False), (1, lower, True, False)):
        lo, hi = domain(c, '1' if nominal else safety, lower=low, inward=True)
        width = hi-lo
        eligible = guard_nominal if nominal else guard[guard[:, 0] >= lo[0]]
        y = evaluate(c, eligible, quotient=quotient)
        picked = []
        for ix in np.argsort(sign*y):
            t = np.divide(eligible[ix]-lo, width, out=np.zeros(c.dim), where=width>0)
            if all(np.linalg.norm(t-q) > .02 for q in picked): picked.append(t)
            if len(picked) >= starts: break
        def obj(t):
            v, g = evaluate(c, lo+width*t, variables=True, quotient=quotient)
            return sign*v[0], sign*g[0]*width
        for t in picked:
            r = minimize(obj, t, jac=True, method='L-BFGS-B', bounds=[(0,1)]*c.dim,
                         options={'maxiter': 100, 'ftol': 2e-14, 'gtol': 1e-10})
            points.append(lo+width*np.clip(r.x, 0, 1))
    return np.asarray(points).reshape(-1, c.dim)


def retained_constraint_points(c, z, lower, final_points):
    """Keep the worst sample points to start later refinements of this circuit.

    Keeps the at most 36 final worst cases and up to 20 samples each with the
    largest output, the smallest h, and the smallest output with unit rounding
    factors. These points do not verify the circuit.
    """
    def finite_unique(points):
        points = np.asarray(points, dtype=float).reshape(-1, c.dim)
        return np.unique(points[np.isfinite(points).all(axis=1)], axis=0)

    final = finite_unique(final_points)[:36]
    guard = finite_unique(z)
    selected = [final]
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        if len(guard):
            y, factor = evaluate(c, guard, both=True)
            indices = np.flatnonzero(np.isfinite(y))
            ordered = indices[np.argsort(y[indices])]
            selected.append(guard[ordered[-20:]])
            indices = np.flatnonzero(np.isfinite(factor))
            selected.append(guard[indices[np.argsort(factor[indices], kind='stable')[:20]]])
        nominal = nominal_points(c, guard, lower)
        y = evaluate(c, nominal)
        indices = np.flatnonzero(np.isfinite(y))
        selected.append(nominal[indices[np.argsort(y[indices])[:20]]])
    points = finite_unique(np.concatenate(selected, axis=0))
    return {'final_adversarial_points': final.tolist(),
            'constraint_points': points.tolist(),
            'constraint_point_columns': ['x']+[f'v{j}' for j in range(c.depth+1)]}


def score(c, z, safety, lower, zn=None, values=False):
    """Return the merit, the sampled lower endpoint, and the constraint violation.

    The violation is the largest amount by which safety*output exceeds one or h
    falls below zero at the samples z. The merit is penalized(r, violation).
    zn holds nominal_points of z, if already computed. values also returns the
    outputs and h at z and the outputs at zn, or None if any is not finite.
    """
    y, factor = evaluate(c, z, both=True); s = float(Fraction(safety))
    if not np.isfinite(y).all() or not np.isfinite(factor).all():
        return (-np.inf, 0., np.inf, None) if values else (-np.inf, 0., np.inf)
    nominal = evaluate(c, nominal_points(c, z, lower) if zn is None else zn)
    if not np.isfinite(nominal).all():
        return (-np.inf, 0., np.inf, None) if values else (-np.inf, 0., np.inf)
    rho = float(np.min(nominal))
    violation = float(max(0., s*np.max(y)-1, -np.min(factor)))
    if values:
        return penalized(rho, violation), rho, violation, (y, factor, nominal)
    return penalized(rho, violation), rho, violation


def penalized(rho, violation):
    """Return the merit of a sampled lower endpoint rho with a constraint violation."""
    return rho-PENALTY*violation


def normalize(c, z, safety):
    """Scale the last row so that safety times the largest sampled output is just below one."""
    y = evaluate(c, z)
    if not np.isfinite(y).all() or y.max() <= 0: return c
    factor = Fraction('0.9999999999')/(Fraction(safety)*Fraction(float(y.max())))
    try: return scale_output(c, factor)
    except OverflowError: return c


def final_row(c, z, safety, lower):
    """Re-solve the last coefficient row by linear programming.

    With the other rows fixed, the output is linear in the last row. The linear
    program maximizes the sampled lower endpoint subject to safety*output <= 1
    and h >= 0 at the samples z.
    """
    t = c.vector()
    f, ff = last_row_features(c, z)
    fn, _ = last_row_features(c, nominal_points(c, z, lower))
    if not all(np.isfinite(value).all() for value in (f,ff,fn)):
        return c, {'status': 'nonfinite'}
    scales = np.maximum(np.max(abs(np.r_[f,ff,fn]), axis=0), 1e-100)
    f, ff, fn = f/scales, ff/scales, fn/scales
    s = float(Fraction(safety))
    A = np.r_[np.c_[f, np.zeros(len(z))], np.c_[-ff, np.zeros(len(z))],
              np.c_[-fn, np.ones(len(fn))]]
    r = solve_lp([0,0,0,-1], A_ub=A,
           b_ub=np.r_[np.full(len(z), 1/s), np.zeros(len(z)+len(fn))],
           bounds=[(None,None)]*4)
    if not r.success:
        return c, {'status': 'lp_failed', 'message': r.message}
    w = r.x[:3]/scales
    t[-3:] = w
    try: return c.with_vector(t), {'status': 'recovered', 'lp_r': float(r.x[-1])}
    except (ValueError, OverflowError): return c, {'status': 'recovery_failed'}


def step_basis(j):
    """Return step directions that whiten the sampled Jacobian, without redundant ones."""
    scales = np.minimum(100., 1/np.maximum(abs(j).max(axis=0), 1e-12))
    _, singular, vt = np.linalg.svd(j*scales/np.sqrt(len(j)), full_matrices=False)
    threshold = np.finfo(float).eps*max(j.shape)*float(singular[0])
    keep = singular > threshold
    singular, vt = singular[keep], vt[keep]
    if not len(singular):
        return np.zeros((j.shape[1],0))
    floor = max(1e-8, 1e-4*float(singular[0]))
    return scales[:,None]*vt.T/np.maximum(singular, floor)


def nonlinear_step(c, z, lower, basis, s, radius, zn=None):
    """Take one sequential quadratic programming step along the directions in basis.

    SLSQP maximizes the merit, the sampled lower endpoint minus PENALTY times
    the violation, within the trust radius. zn holds nominal_points of z, if
    already computed.
    """
    from scipy.optimize import minimize
    if zn is None:
        zn = nominal_points(c, z, lower)
    n, nf = len(z), len(zn)
    steps = basis.shape[1]
    t = c.vector()
    _, rho, violation = score(c, z, s, lower, zn)
    start = np.r_[np.zeros(steps), rho, violation+1e-10]
    cache = {}

    def obj(q):
        return -q[-2]+PENALTY*q[-1], np.r_[np.zeros(steps), -1., PENALTY]

    def constraints(q):
        if 'q' in cache and np.array_equal(q, cache['q']):
            return cache['result']
        trial = c.with_vector(t+basis@q[:-2])
        y, j, factor, jf = evaluate(trial, z, parameters=True, both=True)
        yn, jn = evaluate(trial, zn, parameters=True)
        if not all(np.isfinite(v).all() for v in (y, j, factor, jf, yn, jn)):
            raise FloatingPointError('Nonfinite SQP linearization')
        j, jf, jn = j@basis, jf@basis, jn@basis
        value = np.r_[1-1e-10-s*y+q[-1], factor+q[-1], yn-q[-2]]
        jac = np.r_[np.c_[-s*j,np.zeros(n),np.ones(n)],
                    np.c_[jf,np.zeros(n),np.ones(n)],
                    np.c_[jn,-np.ones(nf),np.zeros(nf)]]
        cache.update(q=q.copy(), result=(value, jac))
        return value, jac

    # SLSQP may try a point outside its box and clip it internally. That normal
    # solver diagnostic is noisy across many restarts; other warnings remain.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Values in x were outside bounds during a minimize step',
                                category=RuntimeWarning)
        r = minimize(obj,start,jac=True,method='SLSQP',
        bounds=[(-radius,radius)]*steps+[(None,None),(0,None)],
        constraints={'type':'ineq','fun':lambda q:constraints(q)[0], 'jac':lambda q:constraints(q)[1]},
        options={'maxiter':SQP_MAXITER,'ftol':1e-11})
    trial = c.with_vector(t+basis@r.x[:-2])
    _, rho, violation = score(trial, z, s, lower, zn)
    r.x[-2:] = [rho, violation]
    return r


def refine(initial, safety, lower, rngseed, initial_points, max_iterations):
    """Optimize a circuit on sample points with a trust region.

    Each iteration takes a sequential quadratic programming step. Periodically,
    the last row is re-solved and local worst cases are added to the samples.
    max_iterations is an upper bound; a collapsed trust region or a nonfinite
    linearization stops early. Returns the best circuit and its diagnostics.
    """
    started = time.monotonic()
    z = guards(initial, safety, lower, rngseed)
    if initial_points is not None:
        extra = np.asarray(initial_points, dtype=float)
        if extra.size:
            z = np.unique(np.concatenate((z, extra), axis=0), axis=0)
    c = normalize(initial, z, safety)
    best = c
    radius = .03
    s = float(Fraction(safety))
    # Nominal points of the samples z, recomputed whenever z changes.
    zn = nominal_points(c, z, lower)
    bestscore = score(best, z, s, lower, zn)[0]
    work = z[np.linspace(0, len(z)-1, min(len(z), 300)).astype(int)]
    accepted = rejected = 0
    trace = []
    reason = 'iteration_limit'
    it = -1
    for it in range(max_iterations):
        if it % 15 == 0:
            extra = adversaries(c, safety, lower, z, guard_nominal=zn)
            z = np.unique(np.r_[z, extra], axis=0)
            zn = nominal_points(c, z, lower)
            work = np.unique(np.r_[work, extra], axis=0)
            bestscore = score(best, z, s, lower, zn)[0]
        merit, rho, violation, values = score(c, z, s, lower, zn, values=True)
        if not np.isfinite(merit):
            reason = 'nonfinite_start'
            break
        # Retain the strongest safety and performance constraints in the SQP
        # working set, while still accepting trials against the full guard set.
        y, factor, yn = values
        indices = np.unique(np.r_[np.argsort(y)[-48:], np.argsort(factor)[:24]])
        nominal_active = zn[np.argsort(yn)[:48]]
        work = np.unique(np.r_[work, z[indices], nominal_active], axis=0)
        if len(work) > 1000:
            work = np.unique(np.r_[work[np.linspace(0, len(work)-1, 400).astype(int)],
                                    z[indices], nominal_active], axis=0)
        if it % 8 == 0:
            trial, _ = final_row(c, work, safety, lower)
            trial = normalize(trial, z, safety)
            trial_merit, trial_rho, trial_violation = score(trial, z, s, lower, zn)
            if trial_merit > merit:
                c = trial
                merit, rho, violation = trial_merit, trial_rho, trial_violation
                if merit > bestscore:
                    best, bestscore = c, merit
        work_nominal = nominal_points(c, work, lower)
        y, j, factor, jf = evaluate(c, work, parameters=True, both=True)
        yn, jn = evaluate(c, work_nominal, parameters=True)
        if not all(np.isfinite(v).all() for v in (y, j, factor, jf, yn, jn)):
            reason = 'nonfinite_linearization'
            break
        basis = step_basis(np.r_[j, jf, jn])
        try:
            r = nonlinear_step(c, work, lower, basis, s, radius, work_nominal)
            if not r.success and not np.isfinite(r.x).all():
                raise ValueError('step failed')
            trial = c.with_vector(c.vector()+basis@r.x[:-2])
            # Rescale the last row before scoring the trial, so that a step that only
            # overshoots the upper bound is not rejected for a scale error that
            # normalization removes. Rescaling keeps the sign of every output.
            trial = normalize(trial, z, safety)
            trial_merit = score(trial, z, s, lower, zn)[0]
        except (ValueError, OverflowError, FloatingPointError):
            radius *= .4
            rejected += 1
            if radius < 1e-10:
                reason = 'subproblem_radius_collapse'
                break
            continue
        predicted = penalized(r.x[-2], r.x[-1])-merit
        accept = bool(trial_merit-merit > max(1e-13, .03*max(predicted, 0)))
        if accept:
            c = trial
            accepted += 1
            if trial_merit > bestscore:
                best, bestscore = c, trial_merit
            if trial_merit-merit > .6*predicted:
                radius = min(.15, radius*1.4)
        else:
            radius *= .5
            rejected += 1
        if it % 5 == 0:
            trace.append(dict(iteration=it, seconds=time.monotonic()-started,
                              r=rho, violation=violation, radius=radius, accepted=accept))
        if radius < 1e-10:
            reason = 'local_trust_radius_plateau'
            break
    final_points = adversaries(best, safety, lower, z, starts=12, guard_nominal=zn)
    z = np.unique(np.r_[z, final_points], axis=0)
    best = normalize(best, z, safety)
    _, rho, violation = score(best, z, safety, lower)
    return best, {'status':'sampled_candidate','sampled_r':rho,
        'sampled_violation':violation,'seconds':time.monotonic()-started,'iterations':it+1,
        'accepted':accepted,'rejected':rejected,'stop_reason':reason,'trace':trace,
        **retained_constraint_points(best,z,lower,final_points)}


# Deterministic enumeration and fixed-work search budgets.

@dataclass(frozen=True)
class Settings:
    """Search budget. The defaults reproduce the included depth-6 circuit, stage1.json."""
    restarts: int = 16
    rounds: int = 300
    finalists: int = 4
    feedback: int = 3
    max_boxes: int = DEFAULT_MAX_BOXES
    tolerance: float = DEFAULT_TOLERANCE
    workers: int = 1

    def __post_init__(self):
        """Check that every count is an integer of at least 1, or 0 for feedback."""
        for name in ('restarts', 'rounds', 'finalists', 'feedback', 'max_boxes', 'workers'):
            value = getattr(self, name)
            minimum = 0 if name == 'feedback' else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer of at least {minimum}')
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('Tolerance must be finite and positive')


def circuit_hash(circuit):
    """Return the SHA-256 hash of the exact coefficients."""
    record = circuit.record()
    exact = {key: record[key] for key in ('model', 'instructions', 'rows_hex')}
    return hashlib.sha256(json.dumps(exact, sort_keys=True).encode()).hexdigest()


def fresh_start(instructions, seed):
    """Return a circuit with random coefficient rows.

    Each row is rescaled so that its values on a grid of inputs in [0, 1] lie
    in [-1, 1], which avoids exponential growth. No saved coefficients are used.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0., 1., 257)
    buffers = [x*x, None]
    rows = []
    for w, p, q, t in instructions:
        a, b = rng.uniform(-2., 2., 2)
        value = a*buffers[p]*buffers[q]+b*buffers[t]
        center = .5*(value.max()+value.min())
        radius = max(.5*(value.max()-value.min()), 1e-4)
        row = [a/radius, b/radius, -center/radius]
        rows.append(row)
        buffers[w] = (value-center)/radius
    # Shift the last row so the initial output is positive before the last row is re-solved.
    rows[-1][2] += 1.1
    return Circuit(instructions, rows)


def _seed(seed, depth, pattern, start):
    """Return the seed of one restart, independent of the budget and the number of workers."""
    return int(np.random.SeedSequence([seed, depth, pattern, start]).generate_state(1)[0])


def _fit(task):
    """Optimize one start from random coefficients."""
    instructions, rngseed, safety, lower, rounds = task
    initial = fresh_start(instructions, rngseed)
    z = guards(initial, safety, lower, rngseed)
    initial, projection = final_row(initial, z, safety, lower)
    initial, conditioning = normalize_states(initial, seed=rngseed, safety=safety)
    candidate, diagnostics = refine(initial, safety, lower, rngseed,
                                    max_iterations=rounds, initial_points=None)
    return candidate, diagnostics, dict(initial_hash=circuit_hash(initial),
                                        projection=projection, conditioning=conditioning)


def _parallel(function, tasks, pool):
    """Map function over tasks, with the worker processes of pool if given, in task order."""
    if not tasks:
        return
    if pool is None:
        yield from map(function, tasks)
    else:
        # Consume in input order: completion timing never selects the winner.
        yield from pool.map(function, tasks)


def _verify_finalist(task):
    """Verify a finalist, refining it between attempts with the verifier's worst cases.

    Returns the evidence of each verification attempt: at most settings.feedback
    + 1, since the cycles stop once one does not improve the verified ratio of
    the lower endpoint to the maximum output.
    """
    candidate, diagnostics, safety, lower, seed, rngseed, si, settings = task
    points = diagnostics['constraint_points']
    evidence = []
    best = -np.inf
    for cycle in range(settings.feedback+1):
        proof = certify(candidate, safety, lower, settings.tolerance, settings.max_boxes)
        item = dict(start=si, cycle=cycle, certificate=proof)
        evidence.append(item)
        r = proof['ratio'] if proof['feasible'] else -np.inf
        improved = not np.isfinite(best) or r > best+FEEDBACK_GAIN*abs(best)
        best = max(best, r)
        if cycle == settings.feedback or (cycle and not improved):
            break
        witnesses = [part['witness'] for part in proof['parts'].values()
                     if part.get('witness') is not None]
        points = np.unique(np.asarray(points+witnesses), axis=0).tolist()
        candidate, _ = normalize_states(candidate, seed=seed, safety=safety)
        candidate, diagnostics = refine(candidate, safety, float(Fraction(lower)),
            rngseed, initial_points=points, max_iterations=settings.rounds)
        points = diagnostics['constraint_points']
        item['feedback_diagnostics'] = diagnostics
    return evidence


def search_stage(lower, safety, depth, *, seed, settings, progress=None, checkpoint=None):
    """Search the circuits of one outer iteration with inputs in [lower, 1].

    Every buffer pattern is optimized from settings.restarts random starts. The
    best distinct candidates over all patterns are verified, and the verified
    circuit with the largest ratio of lower endpoint to maximum output becomes
    the incumbent.
    """
    if settings.workers == 1:
        return _search_stage(lower, safety, depth, seed=seed, settings=settings, pool=None,
                             progress=progress, checkpoint=checkpoint)
    # One set of worker processes serves fitting and verification.
    with ProcessPoolExecutor(max_workers=settings.workers) as pool:
        return _search_stage(lower, safety, depth, seed=seed, settings=settings, pool=pool,
                             progress=progress, checkpoint=checkpoint)


def _search_stage(lower, safety, depth, *, seed, settings, pool, progress, checkpoint):
    """Run search_stage with the worker processes of pool, or in this process if pool is None."""
    ell, s = Fraction(lower), Fraction(safety)
    pattern_list = list(patterns(depth))
    started = time.monotonic()
    out = dict(status='unresolved', lower=str(ell), safety=str(s), depth=depth,
               seed=seed, settings=asdict(settings),
               seed_derivation='NumPy SeedSequence([root_seed, depth, pattern_index, restart_index]) -> uint32',
               pattern_count=len(pattern_list), patterns=[], incumbent=None)

    def emit(event):
        if progress is not None:
            progress(event)

    # Fit all patterns at once, so many workers are useful even with few starts
    # per pattern. The results do not depend on the number of workers.
    tasks, labels, entries, candidates = [], [], {}, {}
    for pi, (bits, instructions) in enumerate(pattern_list):
        entry = dict(bits=bits, starts=[], verifications=[])
        out['patterns'].append(entry)
        entries[pi], candidates[pi] = entry, []
        for si in range(settings.restarts):
            rngseed = _seed(seed, depth, pi, si)
            emit(dict(event='queued', pattern=bits, start=si, seed=rngseed))
            tasks.append((instructions, rngseed, str(s), float(ell), settings.rounds))
            labels.append((pi, bits, si, rngseed))
    for (pi, bits, si, rngseed), (candidate, diagnostics, initialization) in zip(
            labels, _parallel(_fit, tasks, pool)):
        entries[pi]['starts'].append(dict(start=si, seed=rngseed, **initialization,
                                         circuit=candidate.record(), diagnostics=diagnostics))
        emit(dict(event='fitted', pattern=bits, start=si, sampled_r=diagnostics['sampled_r']))
        if checkpoint:
            checkpoint(out)
        if np.isfinite(diagnostics['sampled_r']):
            merit = penalized(diagnostics['sampled_r'], diagnostics['sampled_violation'])
            candidates[pi].append((merit, si, candidate, diagnostics))
    # Verify the best distinct candidates over all patterns, by merit, then pattern and start.
    ranked = sorted(((merit, pi, si, candidate, diagnostics) for pi in sorted(candidates)
                     for merit, si, candidate, diagnostics in candidates[pi]),
                    key=lambda row: (-row[0], row[1], row[2]))
    tasks, labels, seen = [], [], set()
    for _, pi, si, candidate, diagnostics in ranked:
        digest = circuit_hash(candidate)
        if digest in seen:
            continue
        seen.add(digest)
        tasks.append((candidate, diagnostics, str(s), str(ell), seed,
                      _seed(seed, depth, pi, si), si, settings))
        labels.append(pi)
        if len(tasks) == settings.finalists:
            break
    emit(dict(event='verification_started', finalists=len(tasks), workers=settings.workers))
    for pi, evidence in zip(labels, _parallel(_verify_finalist, tasks, pool)):
        bits = pattern_list[pi][0]
        entries[pi]['verifications'].extend(evidence)
        for item in evidence:
            proof = item['certificate']
            # Keep every feasible incumbent, even if later refinement failed.
            if proof['feasible']:
                incumbent = out['incumbent']
                if incumbent is None or proof['ratio'] > incumbent['certificate']['ratio']:
                    out['incumbent'] = dict(pattern=bits, **item)
                    emit(dict(event='incumbent', pattern=bits, ratio=proof['ratio'], status=proof['status']))
        if checkpoint:
            checkpoint(out)
    if out['incumbent'] is not None:
        out['status'] = out['incumbent']['certificate']['status']
    fits = [item['diagnostics'] for entry in out['patterns'] for item in entry['starts']]
    proofs = [item for entry in out['patterns'] for item in entry['verifications']]
    feedback = [item['feedback_diagnostics'] for item in proofs if 'feedback_diagnostics' in item]
    out['work'] = dict(
        refined_starts=sum(len(entry['starts']) for entry in out['patterns']),
        feedback_refinements=len(feedback),
        optimization_rounds=sum(item['iterations'] for item in fits+feedback),
        optimization_task_seconds=sum(item['seconds'] for item in fits+feedback),
        certificates=len(proofs),
        verification_boxes=sum(item['certificate']['boxes_processed'] for item in proofs),
        verification_task_seconds=sum(item['certificate']['seconds'] for item in proofs))
    out['elapsed_seconds'] = time.monotonic()-started
    return out


# Export: buffer scaling for float16 and final verification.

def state_ranges(circuit, safety, upper):
    """Bound every stored value over inputs in [0, upper] by interval arithmetic.

    The inputs are split into RANGE_BINS bins. Each rounding factor ranges over
    [1/safety, safety], and a buffer read twice has the same value in both reads.
    """
    s, upper = Fraction(safety), Fraction(upper)
    edges = np.linspace(0., rational_bounds(upper)[1], RANGE_BINS+1)
    x = I(edges[:-1], edges[1:])
    storage = I(rational_bounds(1/s)[0], rational_bounds(s)[1])
    buffers = [storage*x.square(), None]
    records = []

    def record(value, version, slot):
        lo, hi = float(np.min(value.lo)), float(np.max(value.hi))
        if not np.isfinite([lo, hi]).all():
            raise ArithmeticError('Nonfinite intermediate range')
        records.append(dict(version=version, slot=slot, interval=[lo, hi],
                            max_abs=max(abs(lo), abs(hi))))

    record(buffers[0], 0, 0)
    with np.errstate(over='ignore', invalid='ignore'):
        for j, ((w, p, _, t), (a, b, c)) in enumerate(zip(circuit.instructions, circuit.rows), 1):
            value = (quadratic_range(buffers[p], a, b, c) if p == t
                     else a*buffers[p].square()+b*buffers[t]+c)
            buffers[w] = storage*value
            record(buffers[w], j, w)
    return records


def store_ranges(circuit, safety, upper):
    """Return the store ranges with binary64 and with FP32 coefficients.

    Raises ArithmeticError if a range is not finite, if any store reaches
    STATE_CEILING, or if a hidden store leaves [STORE_MIN, STORE_MAX].
    """
    ranges = state_ranges(circuit, safety, upper)
    with np.errstate(over='ignore', invalid='ignore'):
        rows32 = circuit.rows.astype(np.float32).astype(float)
    if not np.isfinite(rows32).all():
        raise ArithmeticError('FP32 coefficient conversion overflow')
    ranges32 = state_ranges(Circuit(circuit.instructions, rows32), safety, upper)
    if (max(r['max_abs'] for r in ranges+ranges32) >= STATE_CEILING
            or any(not STORE_MIN*(1-1e-10) <= r['max_abs'] <= STORE_MAX*(1+1e-10)
                   for r in ranges[1:-1])):
        raise ArithmeticError('Export exceeds the state-range policy')
    return ranges, ranges32


def _decimal(value):
    """Return a rational as a Decimal."""
    value = Fraction(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def _root_power_two(value, power):
    """Return the root of value of order power, a power of two."""
    while power > 1:
        value = value.sqrt()
        power //= 2
    return value


def scale_candidates(circuit, maxima, initial_scale, final_scale):
    """Yield rescaled circuits, with the most square coefficients equal to +1 first.

    Store j is multiplied by q[j]. Setting q[j] = q[j-1]**2/a[j] makes the
    square coefficient of row j equal to +1; a negative q[j] negates the store,
    which is exact. The square coefficient of every other row is made positive,
    except in the last row, whose positive scale is fixed by the output. A
    linear program in log2|q| proposes scales that keep every store maximum
    within [1/16, 1024], and the rows are then computed with 120 decimal
    digits. The caller verifies each candidate.
    """
    from scipy.optimize import linprog
    n = circuit.depth
    # The last square coefficient can be made +1 only if it is positive.
    eligible = [j for j, row in enumerate(circuit.rows) if row[0] > 0 or (row[0] < 0 and j < n-1)]
    if any(not np.isfinite(m) or m <= 0 for m in maxima):
        raise ArithmeticError('State bounds must be finite and positive')
    z0, zn = math.log2(initial_scale), math.log2(final_scale)
    limits = [(math.log2(1/16)-math.log2(m), 10-math.log2(m)) for m in maxima[:-1]]
    limits.append((zn, zn))
    target = [min(high, max(low, 0.)) for low, high in limits]
    inequality = np.zeros((2*n, 2*n))
    rhs = np.empty(2*n)
    for j in range(n):
        inequality[2*j, j], inequality[2*j, n+j] = 1, -1
        inequality[2*j+1, j], inequality[2*j+1, n+j] = -1, -1
        rhs[2*j], rhs[2*j+1] = target[j], -target[j]
    for count in range(len(eligible), -1, -1):
        choices = []
        for mask in combinations(eligible, count):
            equality = np.zeros((count, 2*n))
            values = []
            for k, j in enumerate(mask):
                equality[k, j] = 1
                if j:
                    equality[k, j-1] = -2
                values.append(-math.log2(abs(circuit.rows[j, 0]))+(2*z0 if j == 0 else 0))
            solved = linprog(np.r_[np.zeros(n), np.ones(n)], A_ub=inequality, b_ub=rhs,
                A_eq=equality if count else None, b_eq=values if count else None,
                bounds=limits+[(0, None)]*n, method='highs')
            if solved.success:
                choices.append((float(solved.fun), mask, solved.x[:n]))
        for distance, mask, logs in sorted(choices, key=lambda item: (item[0], item[1])):
            with localcontext() as context:
                context.prec = 120
                selected = {j+1 for j in mask}
                starts = [0]+[j for j in range(1, n+1) if j not in selected]
                scales = [None]*(n+1)
                valid = True
                for index, start in enumerate(starts):
                    end = starts[index+1]-1 if index+1 < len(starts) else n
                    factors, powers = {start: Decimal(1)}, {start: 1}
                    for j in range(start+1, end+1):
                        factors[j] = factors[j-1]**2/_decimal(circuit.rows[j-1, 0])
                        powers[j] = 2*powers[j-1]
                    low, high = Decimal(0), Decimal('Infinity')
                    for j in range(max(start, 1), min(end, n-1)+1):
                        magnitude = _decimal(maxima[j-1])*abs(factors[j])
                        low = max(low, _root_power_two(1/(16*magnitude), powers[j]))
                        high = min(high, _root_power_two(1024/magnitude, powers[j]))
                    if start == 0:
                        anchor = _decimal(initial_scale)
                    elif end == n:
                        if factors[end] <= 0:  # the final scale is positive
                            valid = False
                            break
                        anchor = _root_power_two(_decimal(final_scale)/factors[end], powers[end])
                    else:
                        anchor = min(high, max(low, Decimal(2)**Decimal(float(logs[start-1]))))
                    if low > high or anchor < low*(1-Decimal('1e-90')) or anchor > high*(1+Decimal('1e-90')):
                        valid = False
                        break
                    # The sign of the first scale of a segment makes its square coefficient
                    # positive; later scales take the sign of their row, since even powers
                    # of the anchor are positive. The last store's scale is fixed.
                    if 0 < start < n and circuit.rows[start-1, 0] < 0:
                        anchor = -anchor
                    for j in range(start, end+1):
                        scales[j] = factors[j]*anchor**powers[j]
                if not valid or abs(scales[-1]/_decimal(final_scale)-1) > Decimal('1e-90'):
                    continue
                scales[-1] = _decimal(final_scale)
                # Convert every actual source version, including older addends.
                versions, rows = [0, None], []
                for j, ((w, p, q, t), row) in enumerate(zip(circuit.instructions, circuit.rows), 1):
                    a, b, c = map(_decimal, row)
                    new = [a*scales[j]/(scales[versions[p]]*scales[versions[q]]),
                           b*scales[j]/scales[versions[t]], c*scales[j]]
                    if j in selected:
                        new[0] = Decimal(1)
                    rows.append([float(v) for v in new])
                    versions[w] = j
                if not np.isfinite(rows).all():
                    continue
                yield Circuit(circuit.instructions, rows), dict(
                    scales=[str(v) for v in scales], unit_square_steps=[j+1 for j in mask],
                    conditioning_distance=distance, reconstruction_decimal_digits=120)


def reusable_certificate(proof, circuit, safety, lower, tolerance, max_boxes):
    """Return True if proof is a feasible certify result for exactly this circuit and these settings."""
    record = circuit.record()
    return bool(proof is not None and proof.get('feasible')
                and proof['circuit']['instructions'] == record['instructions']
                and proof['circuit']['rows_hex'] == record['rows_hex']
                and proof['safety'] == str(Fraction(safety))
                and proof['lower_endpoint_domain'] == [str(Fraction(lower)), '1']
                and proof['output_cap'] is None and proof['requested_accuracy'] == tolerance
                and proof.get('max_boxes') == max_boxes)


def export_stage(circuit, lower, safety, *, upper, center, tolerance, max_boxes, certificate=None):
    """Rescale a circuit searched on [lower/upper, 1] to inputs in [lower, upper].

    The first store absorbs the input scaling, q[0] = upper**2. The last store
    absorbs the scaling of the verified maximum output to one, the optional
    centering 2/(1 + ratio), and 1/upper, so that the circuit runs without
    additional scaling. The intermediate stores are
    rescaled by scale_candidates. Returns the first candidate whose store
    ranges, also with FP32 coefficients, pass store_ranges and that certify verifies.
    certificate, the search's verification of circuit, is reused when it was
    made for exactly this circuit and these settings.
    """
    lower, upper, safety = map(Fraction, (lower, upper, safety))
    # The search has usually verified this exact circuit already.
    original = (certificate if reusable_certificate(certificate, circuit, safety, lower/upper, tolerance, max_boxes)
                else certify(circuit, safety, lower/upper, tolerance, max_boxes))
    if not original['feasible']:
        raise ArithmeticError('Input circuit could not be verified')
    # The search does not rescale circuits; the verified maximum output sets the scale here.
    original_maximum = Fraction(original['robust_maximum_bracket'][1])
    multiplier = 2/(1+Fraction(original['ratio'])) if center else Fraction(1)
    output_cap = rational_bounds(multiplier)[1]
    before = state_ranges(circuit, safety, 1)
    rejected = []
    output_scale_correction = Fraction(1)
    # Rounding a safe circuit can raise its maximum slightly. Only a proven
    # upper bound may trigger a conservative correction, followed by a recheck.
    for _ in range(3):
        corrected = False
        final = multiplier*output_scale_correction/(upper*original_maximum)
        for exported, scaling in scale_candidates(circuit, [r['max_abs'] for r in before[1:]], upper**2, final):
            try:
                ranges, ranges32 = store_ranges(exported, safety, upper)
            except ArithmeticError as exc:
                rejected.append(dict(unit_square_steps=scaling['unit_square_steps'], reason=str(exc)))
                continue
            proof = certify(exported, safety, lower, tolerance, max_boxes, upper=upper, output_upper=output_cap)
            if proof['feasible']:
                return dict(status='verified', circuit=exported.record(),
                    input_interval_exact=[str(lower), str(upper)],
                    output_interval=[proof['r'], output_cap],
                    output_interval_exact=[str(Fraction(proof['r'])), str(Fraction(output_cap))],
                    safety=str(safety), centered=bool(center), centering_multiplier_exact=str(multiplier),
                    verification_settings=dict(tolerance=tolerance, max_boxes=max_boxes),
                    verification=proof, original_verification=original,
                    scaling=scaling | dict(original_state_ranges=before, exported_state_ranges=ranges,
                        fp32_coefficient_state_ranges=ranges32, range_bins=RANGE_BINS,
                        state_ceiling=STATE_CEILING, rounding_output_scale_correction_exact=str(output_scale_correction),
                        rejected_candidates=rejected))
            maximum = proof.get('robust_maximum_bracket', [None, None])[1]
            if maximum is not None and maximum > output_cap and proof['r'] > 0:
                output_scale_correction *= Fraction(output_cap)/Fraction(maximum)*ROUNDING_MARGIN
                corrected = True
                break
            rejected.append(dict(unit_square_steps=scaling['unit_square_steps'],
                                 reason='Exported coefficients could not be certified', verification=proof))
        if not corrected:
            break
    raise ArithmeticError('No scaled circuit passed coefficient and state-range verification')


def circuit_summary(result):
    """Return the summary written to PREFIX.json, with every operand read before overwriting."""
    circuit = Circuit.from_record(result['circuit'])
    names = ('A', 'B')
    steps = []
    for j, ((w, p, q, t), row) in enumerate(zip(circuit.instructions, circuit.rows), 1):
        steps.append(dict(step=j, write=names[w], square=names[p], addend=names[t],
            a=float(row[0]), b=float(row[1]), c=float(row[2]),
            operation=f'{names[w]} = a * ({names[p]} @ {names[q]}) + b * {names[t]} + c * I'))
    final = names[circuit.instructions[-1][0]]
    return dict(format=SUMMARY_FORMAT, status='verified', depth=circuit.depth,
        input_interval=[float(Fraction(v)) for v in result['input_interval_exact']],
        output_interval=result['output_interval'], safety_factor=result['safety'],
        centered=result['centered'],
        iteration=dict(
            initialize='A = X.T @ X when X has more rows than columns; otherwise A = X @ X.T. Allocate B with the same shape as A.',
            steps=steps,
            finish=f'Y = X @ {final} when X has more rows than columns; otherwise Y = {final} @ X.',
            note='@ is matrix multiplication; I is the identity matrix. Read all operands before overwriting a buffer. All scaling and optional centering are already included in these coefficients.'),
        verification='The lower endpoint is verified for the exact circuit. The upper endpoint and buffer ranges are verified when every stored value is multiplied by a rounding factor in [1/safety, safety]; this does not model every GPU rounding operation.',
        verification_settings=result['verification_settings'],
        exact_intervals=dict(input=result['input_interval_exact'], output=result['output_interval_exact']))


def read_summary(summary):
    """Read a verified circuit from a summary written by circuit_summary."""
    if summary.get('status') != 'verified':
        raise ValueError('Summary is not a verified circuit')
    instructions, rows = [], []
    for step in summary['iteration']['steps']:
        w, p, t = ('AB'.index(step[key]) for key in ('write', 'square', 'addend'))
        instructions.append((w, p, p, t))
        rows.append([step['a'], step['b'], step['c']])
    return dict(status='verified', circuit=Circuit(instructions, rows).record(),
                input_interval_exact=summary['exact_intervals']['input'],
                output_interval_exact=summary['exact_intervals']['output'],
                verification_settings=summary['verification_settings'], safety=summary['safety_factor'])


# Standalone command-line interface and reproducibility records.

def metadata():
    """Return the Python, platform, package, and source versions for the details file."""
    packages = {}
    for name in ('numpy', 'scipy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(),
                platform=platform.platform(), machine=platform.machine(),
                numpy_configuration=np.show_config(mode='dicts'),
                packages=packages,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(ROOT.glob('*.py'))},
                threads={key: os.environ.get(key) for key in THREAD_VARIABLES})


def to_json(record):
    """Return the JSON text written for a record."""
    return json.dumps(record, indent=2, allow_nan=False)+'\n'


def write(path, record):
    """Write a JSON record to a new file; an existing file is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(to_json(record))


def save_partial(path, record):
    """Replace the details file atomically to record progress; runs cannot resume from it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(to_json(record), encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def load_case(path):
    """Return the verified result of a summary (PREFIX.json) or details (PREFIX.details.json) file."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    saved = read_summary(payload) if payload.get('format') == SUMMARY_FORMAT else payload.get('result')
    if not saved or saved.get('status') != 'verified':
        raise ValueError('Input has no verified circuit')
    return saved


def check_export(result, tolerance, max_boxes):
    """Verify a saved circuit again: its interval and store ranges, also with FP32 coefficients."""
    circuit = Circuit.from_record(result['circuit'])
    lower, upper = result['input_interval_exact']
    claimed_lower, claimed_upper = map(Fraction, result['output_interval_exact'])
    proof = certify(circuit, result['safety'], lower, tolerance, max_boxes,
                    upper=upper, output_upper=claimed_upper)
    try:
        ranges, ranges32 = store_ranges(circuit, result['safety'], upper)
    except ArithmeticError:
        ranges = ranges32 = None
    valid = bool(proof['feasible'] and Fraction(proof['r']) >= claimed_lower and ranges is not None)
    return dict(verified=valid, certificate=proof, state_ranges=ranges,
                fp32_coefficient_state_ranges=ranges32)


def main(argv=None):
    """Run the search or verify command and return the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ['--help']
    if argv[0] not in ('search', 'verify', '-h', '--help'):
        argv.insert(0, 'search')
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    search = commands.add_parser(
        'search', help='Search, scale, and verify one circuit (the default command)',
        description='Search the circuit of one outer iteration, scale its buffers, and verify it.')
    verify = commands.add_parser(
        'verify', help='Verify a saved circuit and its interval',
        description='Verify a saved circuit (PREFIX.json or PREFIX.details.json) and its interval.')
    for command in (search, verify):
        command.add_argument('--tolerance', type=float, default=None,
                             help='Requested verification accuracy (default: 1e-10 for search, '
                                  'the saved setting for verify)')
        command.add_argument('--max-boxes', type=int, default=None,
                             help=f'Subdivision budget per extremum (default: {defaults.max_boxes} '
                                  'for search, the saved setting for verify)')
        command.add_argument('--verbose', action='store_true', help='Print intermediate progress')
    verify.add_argument('--input', type=Path, required=True, help='Saved circuit file')
    search.add_argument('--depth', type=int, required=True, help='Number of inner iterations in the circuit')
    search.add_argument('--output', type=Path, required=True,
                        help='Write PREFIX.json and PREFIX.details.json; existing files are never overwritten')
    search.add_argument('--lower', help='Input lower endpoint (default .001)')
    search.add_argument('--upper', help='Input upper endpoint (default 1)')
    search.add_argument('--safety', help='Safety factor s of the rounding factors (default 1.005)')
    search.add_argument('--from', dest='previous', type=Path,
                        help='Use the verified output interval and safety factor of a saved circuit as the input; '
                             'cannot be combined with --lower, --upper, or --safety')
    search.add_argument('--center', action='store_true',
                        help='Center the final interval at one; use it for the last outer iteration')
    search.add_argument('--seed', type=int, default=0, help='Nonnegative random seed (default 0)')
    search.add_argument('--workers', type=int, default=1,
                        help='CPU workers, each with one numerical-library thread (default 1)')
    search.add_argument('--restarts', type=int, help=f'Restarts per buffer pattern (default {defaults.restarts})')
    search.add_argument('--rounds', type=int, help=f'Optimization rounds per refinement (default {defaults.rounds})')
    search.add_argument('--finalists', type=int,
                        help=f'Best distinct candidates verified, over all buffer patterns (default {defaults.finalists})')
    search.add_argument('--feedback', type=int,
                        help=f'Most additional optimization and verification cycles; they stop when the verified '
                             f'ratio stops improving (default {defaults.feedback})')
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify':
            saved = load_case(args.input)
            settings = saved['verification_settings']
            checked = check_export(
                saved, settings['tolerance'] if args.tolerance is None else args.tolerance,
                settings['max_boxes'] if args.max_boxes is None else args.max_boxes)
            if args.verbose:
                print(json.dumps(checked, allow_nan=False))
            print('Verified saved circuit and interval.' if checked['verified']
                  else 'Verification failed for the saved circuit and interval.')
            return 0 if checked['verified'] else 2
        if args.depth < 1 or args.seed < 0:
            raise ValueError('Require --depth >= 1 and --seed >= 0')
        summary_path = Path(str(args.output)+'.json')
        details_path = Path(str(args.output)+'.details.json')
        if any(p.exists() for p in (summary_path, details_path, Path(str(details_path)+'.tmp'))):
            parser.error('Output already exists; choose a new --output prefix')
        lower, upper, safety = args.lower or LOWER, args.upper or '1', args.safety or SAFETY
        dependency = None
        if args.previous:
            if args.lower is not None or args.upper is not None or args.safety is not None:
                raise ValueError('--from supplies the interval and safety factor; '
                                 'do not also specify --lower, --upper, or --safety')
            previous = load_case(args.previous)
            checked = check_export(previous, previous['verification_settings']['tolerance'],
                                   previous['verification_settings']['max_boxes'])
            if not checked['verified']:
                raise ValueError('The --from circuit failed verification')
            lower, upper = previous['output_interval_exact']
            safety = previous['safety']
            dependency = dict(file=str(args.previous), sha256=hashlib.sha256(args.previous.read_bytes()).hexdigest(),
                              circuit_sha256=circuit_hash(Circuit.from_record(previous['circuit'])),
                              verification=checked)
        lower, upper, safety = map(Fraction, (lower, upper, safety))
        if not 0 < lower <= upper or safety < 1:
            raise ValueError('Require 0 < lower <= upper and safety >= 1')
        overrides = {item.name: getattr(args, item.name, None) for item in fields(Settings)}
        settings = Settings(**{key: value for key, value in overrides.items() if value is not None})
        details = dict(format='two-buffer-search-details', status='running',
            inputs=dict(lower=str(lower), upper=str(upper), safety=str(safety), center=args.center),
            parameters=asdict(settings), dependency=dependency,
            reproducibility=dict(command={key: str(value) if isinstance(value, Path) else value
                                          for key, value in vars(args).items()}, software=metadata()))
        write(details_path, details)
    except (ValueError, ArithmeticError, KeyError, OSError) as exc:
        parser.error(str(exc))

    def progress(event):
        if args.verbose:
            print(json.dumps(event, allow_nan=False), flush=True)

    last_checkpoint = -math.inf

    def checkpoint(result):
        nonlocal last_checkpoint
        details['search'] = result
        # Large searches retain all history, but need not rewrite it after
        # every restart. This timer controls I/O only, never search decisions.
        now = time.monotonic()
        if now-last_checkpoint >= 15.:
            save_partial(details_path, details)
            last_checkpoint = now

    try:
        result = search_stage(str(lower/upper), str(safety), args.depth, seed=args.seed,
                              settings=settings, progress=progress, checkpoint=checkpoint)
        details['search'] = result
        if result['status'] != 'verified':
            raise ArithmeticError('No verified circuit found within this search budget')
        circuit = Circuit.from_record(result['incumbent']['certificate']['circuit'])
        progress(dict(event='scaling_and_verifying_export'))
        exported = export_stage(circuit, lower, safety, upper=upper, center=args.center,
                                certificate=result['incumbent']['certificate'],
                                tolerance=settings.tolerance, max_boxes=settings.max_boxes)
        details['result'] = exported
        summary = circuit_summary(exported)
        details['summary_sha256'] = hashlib.sha256(to_json(summary).encode()).hexdigest()
        details['status'] = 'verified'
        save_partial(details_path, details)
        # Write PREFIX.json only after verification succeeds.
        write(summary_path, summary)
        lo, hi = exported['output_interval']
        print(f'Verified output interval: [{lo:.12g}, {hi:.12g}]; lower/upper = {lo/hi:.12g}')
        print(f'Summary: {summary_path}\nDetails: {details_path}')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        details['status'] = 'failed'
        details['error'] = f'{type(exc).__name__}: {exc}'
        save_partial(details_path, details)
        print(f'No verified circuit exported: {exc}', file=sys.stderr)
        print(f'Details: {details_path}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
