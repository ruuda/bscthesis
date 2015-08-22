# Copyright 2015 Ruud van Asseldonk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of version 3 of the GNU General Public License as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from math import atan2, copysign, cos, exp, floor, log, pi, sin, sqrt
from sympy import diff, lambdify, symbols
import numpy as np
import random

### Auxillary functions, vector math ###
########################################

# Composes two functions of one variable.
def compose(f1, f2):
    return lambda t: f1(f2(t))


# Appends the list y to every element of xs. Tuple elements are converted to lists.
def augment(xs, y):
    return [[a for a in x] + y for x in xs]


# Prepends item to the generator.
def cons(item, seq):
    yield item
    for x in seq:
        yield x


# Concatenates two generators.
def cat(seq_a, seq_b):
    for x in seq_a:
        yield x
    for x in seq_b:
        yield x


# Returns the average of two points.
def mid(v, w):
    return [(vi + wi) / 2.0 for (vi, wi) in zip(v, w)]


# Adds two vectors.
def add(v, w):
    return [(vi + wi) for (vi, wi) in zip(v, w)]

# Subtracts two vectors.
def sub(v, w):
    return [(vi - wi) for (vi, wi) in zip(v, w)]


# Returns the z-component of the cross-product of two vectors.
def cross_z(v, w):
    return v[0] * w[1] - v[1] * w[0]


# Retuns the Euclidean norm of a vector.
def eucl_norm(v):
    return sqrt(sum(vi * vi for vi in v))


# Sets the length of the vector to unit length.
def normalise(v):
    norm = eucl_norm(v)
    return [vi / norm for vi in v]


### Hopf-related functions ###
##############################

# Converts a point on S2 to a point on P1(C).
def cartesian_to_homogeneous(x):
    if x[0] == -1:
        return [0, 0, 1, 0]
    else:
        return [0, 1 + x[0], x[1], x[2]]


# Parametrises the fibre above z in P1(C) as array in C2.
# Returns a function of t. For (m, n) = (1, 1) we get the
# fibres of the Hopf map.
def fibre_above(z, m, n):
    # Choose a normalised representant.
    norm = eucl_norm(z)
    z = [zi / norm for zi in z]

    # Convert to polar coordinates on C.
    # Note that the argument order of atan2 is y, x!
    arg1 = atan2(z[1], z[0])
    mod1 = eucl_norm(z[0:2])
    arg2 = atan2(z[3], z[2])
    mod2 = eucl_norm(z[2:4])

    # Computes a point on the fibre, in Cartesian coordinates on C2.
    return lambda t: [mod1 * cos(arg1 + 2 * pi * t * n),
                      mod1 * sin(arg1 + 2 * pi * t * n),
                      mod2 * cos(arg2 + 2 * pi * t * m),
                      mod2 * sin(arg2 + 2 * pi * t * m)]


# Apply the stereographic projection S3 -> R3.
def project_stereo(x):
    f = 1.0 / (1 - x[0])
    return [xi * f for xi in x[1:]]


# Apply the inverse stereographic projection R3 -> S3.
def inverse_project_stereo(x):
    x_sqr = sum([x_i ** 2 for x_i in x])
    pr_0 = (x_sqr - 1) / (x_sqr + 1)
    pr_n = [2 * x_i / (x_sqr + 1) for x_i in x]
    return [pr_0] + pr_n


# The Hopf map from S^3 to S^2 (considered as subsets of R^4 and R^3). The
# formula is based on quaternion multiplication, where q = (a + bi + cj + dk),
# and hopf(q) = q^{-1} i q.
def hopf(x):
    [a, b, c, d] = x
    i_coef = a ** 2 + b ** 2 - c ** 2 - d ** 2
    j_coef = 2 * (b * c - a * d)
    k_coef = 2 * (a * c + b * d)
    return [i_coef, j_coef, k_coef]


# Computes the pullback of omega by f, where omega is a two-form on R^3 and
# f: R^3 -> R^3. x = (x1, x2, x3) is a basis for the tangent space.
def pullback(x, f, omega):
    # df[i][j] is \frac{\partial f_i}{\partial x_j}.
    df = [[diff(f(x)[i], x[j]) for j in [0, 1, 2]] for i in [0, 1, 2]]

    # coef[i] is omega_i after f.
    coef = omega(f(x))

    ijk = list(zip([0, 1, 2], [1, 2, 0], [2, 0, 1]))

    return [sum([coef[p] * (df[q][j] * df[r][k] - df[q][k] * df[r][j])
                 for (p, q, r) in ijk])
            for (i, j, k) in ijk]


# Returns an orthographic projection R3 -> R3.
# Assumes theta rotates around the x1 axis.
def orthographic_projection(phi, theta):
    cp = cos(phi)
    sp = sin(phi)

    rot_elevation = np.matrix([[cp, 0, -sp],
                               [ 0, 1,   0],
                               [sp, 0,  cp]]);

    ct = cos(theta)
    st = sin(theta)
    rot_azimuth = np.matrix([[1,  0,   0],
                             [0, ct, -st],
                             [0, st,  ct]]);

    rot = rot_elevation * rot_azimuth

    pr = lambda v: np.dot(rot, v).tolist()[0]
    pr.phi = phi
    pr.theta = theta
    return pr


# Parametrises the projection of the fibre above x (given in Cartesian coordinates on S2).
# Returns a function of t.
def projected_fibre_from_cartesian(x):
    hom_coord = cartesian_to_homogeneous(x)
    fibre = fibre_above(hom_coord, 1, 1)
    return compose(project_stereo, fibre)


# Converts spherical coordinates to cartesian coordinates.
# Expected format is phi (angle with x2x3 plane), theta (angle around x1 axis).
def spherical_to_cartesian(s):
    [phi, theta] = s
    return [sin(phi), cos(phi) * cos(theta), cos(phi) * sin(theta)]


# Paramatrises the projection of the fibre above x (given in spherical coordinates on S2).
# Expected format is phi (angle with x2x3 plane), theta (angle around x1 axis).
# Returns a function of t.
def projected_fibre_from_spherical(s):
    x = spherical_to_cartesian(s)
    return projected_fibre_from_cartesian(x)


# Parametrises the degenerate fibre with constant velocity and given bounds.
def projected_fibre_from_degenerate(min_x1, max_x1):
    return lambda t: [min_x1 + t * (max_x1 - min_x1), 0, 0]


# Parametrises the fibre through x such that f(0) = x.
def projected_fibre_through(x):
    hom_coord = inverse_project_stereo(x)
    fibre = fibre_above(hom_coord, 1, 1)
    return compose(project_stereo, fibre)


# Parametrises the fibre through x such that f(0) = x.
def projected_knot_fibre_through(x, m, n):
    hom_coord = inverse_project_stereo(x)
    fibre = fibre_above(hom_coord, m, n)
    return compose(project_stereo, fibre)


# Ensures that the curve f(0) through f(1) lies inside the box.
def truncate_fibre(fibre, box):
    def is_inside(x):
        pos = all(xi <  bi for (xi, bi) in zip(x, box))
        neg = all(xi > -bi for (xi, bi) in zip(x, box))
        return pos and neg

    # Make an initial guess at where the fibre leaves the box.
    t0, t1 = 0.0, 0.0
    for i in range(0, 100):
        t0 = -i / 100.0
        if not is_inside(fibre(t0)):
            break
    for i in range(0, 100):
        t1 = i / 100.0
        if not is_inside(fibre(t1)):
            break

    # If the fibre fits inside the box we need not change anything.
    if t0 == -0.99 and t1 == 0.99:
        return fibre

    # Refine the boundary by binary search.
    t0a, t0b = t0 + 0.01, t0
    t1a, t1b = t1 - 0.01, t1
    for _ in range(0, 100):
        t0, t1 = (t0a + t0b) / 2.0, (t1a + t1b) / 2.0
        if is_inside(fibre(t0)):
            t0a = t0
        else:
            t0b = t0
        if is_inside(fibre(t1)):
            t1a = t1
        else:
            t1b = t1

    t0, t1 = t0a, t1a
    delta = t1 - t0
    return lambda t: fibre(t0 + delta * t)


### Plotting-related functions ###
##################################

# Utility for finding plot bounds.
def binsearch_bounds(f, tz):
    t0 = 0.0
    top = f(t0)[0] > tz
    if not top:
        while f(t0)[0] < tz:
            t0 = t0 + 0.05
    while f(t0)[0] > tz:
        t0 = t0 + 0.05

    t1, t0, = t0, t0 - 0.05
    for i in range(0, 100):
        ti = (t0 + t1) / 2.0
        if f(ti)[0] > tz:
            t0 = ti
        else:
            t1 = ti

    print('t = {0}, p = {1}'.format(t0, f(t0)))


# Produces a set of points to sample at, given a function [0, 1] -> R3.
# And a projection function R3 -> R3, where the first coordinate comes out of the screen.
def find_samples(f, resolution):
    # Initially we try this subdivision. We can abuse the fact that the
    # fibres we plot are circular: it means that no weird cusps will appear,
    # and everything will become smoother.
    samples = [0.0, 0.5, 1.0]

    def is_smooth(p0, p1, p2):
        d1 = sub(p1, p0)[1:3]
        d2 = sub(p2, p1)[1:3]
        # The cross product is a good measure for smoothness; it is large
        # if both vectors are long (meaning that we have a coarse subdivision),
        # and it is large if the angle p0 -- p1 -- p2 is large. For small
        # lengths and small angles it is small.
        cp = abs(cross_z(d1, d2))
        n1 = eucl_norm(d1)
        n2 = eucl_norm(d2)

        # Straight lines coming out of the screen are a special case.
        if n1 * n2 == 0:
            return True

        # Decide based on the length of the pieces and the angle between them.
        return (n1 + n2) * (cp / (n1 * n2) + 0.5) < resolution

    # Keep on subdividing until every piece is smooth enough.
    k = 0
    t0, t2 = samples[0:2]
    p0, p2 = f(t0), f(t2)
    while True:
        t1 = (t0 + t2) / 2.0
        p1 = f(t1)
        if is_smooth(p0, p1, p2):
            k = k + 1
            if k == len(samples) - 1:
                break
            t0, t2 = t2, samples[k + 1]
            p0, p2 = p2, f(t2)
        else:
            samples.insert(k + 1, t1)
            t2, p2 = t1, p1

    return samples


# Samples the unit interval [0, 1] in (samples + 1) steps.
def interval_closed(samples):
    return [t / float(samples) for t in range(0, samples + 1)]


# Samples the unit interval [0, 1) in (samples) steps.
def interval_open(samples):
    return [t / float(samples) for t in range(0, samples)]


# Finds a tangent vector of f at f(t) by taking smaller intervals until the
# tangent vector converges enough.
def get_tangent_vector(f, t, tstep):
    p = f(t)
    tau0 = [0, 0]
    tau1 = sub(f(t + tstep), f(t - tstep))

    while True:
        if eucl_norm(tau1) == 0.0:
            return tau0

        tau1 = normalise(tau1)
        if eucl_norm(sub(tau0, tau1)) < 0.00001:
            return tau1
        else:
            tstep = tstep / 2.0
            tau0 = tau1
            tau1 = sub(f(t + tstep), f(t - tstep))


# Returns four points of a cubic Bezier that are a good approximation to the curve f.
def make_bezier(f, t0, t1):
    tm = (t0 + t1) * 0.5
    p0 = f(t0)
    pm = f(tm)
    p1 = f(t1)
    tau0 = get_tangent_vector(f, t0, t1 - t0)
    taum = get_tangent_vector(f, tm, t1 - t0)
    tau1 = get_tangent_vector(f, t1, t1 - t0)

    denom0 = taum[0] * tau0[1] - taum[1] * tau0[0]
    denom1 = taum[0] * tau1[1] - taum[1] * tau1[0]

    # Straight lines are a degenerate case.
    if abs(denom0 * denom1) < 0.0001:
        return [p0, mid(p0, pm), mid(p1, pm), p1]

    # Fit a Bezier curve through (p0, pm, p1) that has the correct tangent
    # vector at every point.
    r0 = (1.0 / 3.0) * (taum[0] * (5 * p0[1] - p1[1] - 4 * pm[1]) - \
                        taum[1] * (5 * p0[0] - p1[0] - 4 * pm[0])) / denom0
    r1 = (1.0 / 3.0) * (taum[0] * (5 * p1[1] - p0[1] - 4 * pm[1]) - \
                        taum[1] * (5 * p1[0] - p0[0] - 4 * pm[0])) / denom1
    q0 = sub(p0, [tau * r0 for tau in tau0])
    q1 = sub(p1, [tau * r1 for tau in tau1])

    return [p0, q0, q1, p1]


# Formats a 2D coordinate.
def format_2d(x):
    return '({0:0.6f}, {1:0.6f})'.format(x[0], x[1])


# Converts a segment to TikZ draw commands.
def draw_2d(segment, front, back):
    # Add a tiny bit of overshoot because curves that align precisely render
    # with a stripe of white artefact in between.
    def overshoot(t):
        sh0 = [(pi - qi) * t for (pi, qi) in zip(segment[0], segment[1])]
        sh1 = [(pi - qi) * t for (pi, qi) in zip(segment[3], segment[2])]
        return [add(segment[0], sh0)] + segment[1:3] + [add(segment[3], sh1)]

    sstr_back  = [format_2d(pt) for pt in overshoot(0.02)]
    sstr_front = [format_2d(pt) for pt in overshoot(0.04)]
    return ('\\draw[{0}] {1} .. controls {2} and {3} .. {4};'.format(back,  *sstr_back) + '\n'
            '\\draw[{0}] {1} .. controls {2} and {3} .. {4};'.format(front, *sstr_front))
    #       '\\draw {0} circle (1pt);'.format(sstr_front[0])) # For debug printing control points.


# Writes all items to the file, one per line.
def write_items(fname, items):
    with open(fname, 'w') as outfile:
        for item in items:
            print(item, file = outfile)


# Writes the coordinates to the file as a table.
def write_coordinates(fname, xs):
    write_items(fname, (' '.join(format(xi, '.4f') for xi in x) for x in xs))


# Generates segments to draw from the given curves.
def generate_raw_2d_segments(resolution, curves):
    def make_full_segments(curve):
        f, *tail = curve
        f_pr = lambda t: f(t)[1:3] # Project onto the x2x3-plane.
        f_tr = lambda t: f(t)[0]   # Distance along the x1-plane.
        samples = find_samples(f, resolution)
        segment_times = list(zip(samples, samples[1:]))
        segments = [make_bezier(f_pr, t0, t1) for (t0, t1) in segment_times]
        depths = [f_tr((t0 + t1) * 0.5) for (t0, t1) in segment_times]
        positions = [[f(t0), f(t1)] for (t0, t1) in segment_times]
        return augment(zip(depths, segments, positions), tail)

    curve_segments = [make_full_segments(curve) for curve in curves]
    segments = [s for segments in curve_segments for s in segments]
    return sorted(segments, key = lambda s: s[0])


# Generates TikZ draw instructions given a list of [f(t), front, back] elements.
def generate_raw_draw_2d(resolution, curves):
    x1_sorted = generate_raw_2d_segments(resolution, curves)
    return (draw_2d(segment, front, back) for [_, segment, _, front, back] in x1_sorted)


# Writes TikZ draw instructions to the file given a list of [f(t), front, back] elements.
def write_raw_draw_2d(fname, resolution, curves):
    write_items(fname, generate_raw_draw_2d(resolution, curves))


# Generates the draw instructions for drawing an axis box.
def generate_raw_box_2d(dimensions, projection, style):
    # down = d, up = u, left = l, right = r, back = b, front = f
    dlb = [-dimensions[0], -dimensions[1],  dimensions[2]]
    dlf = [-dimensions[0], -dimensions[1], -dimensions[2]]
    drb = [-dimensions[0],  dimensions[1],  dimensions[2]]
    drf = [-dimensions[0],  dimensions[1], -dimensions[2]]
    ulb = [ dimensions[0], -dimensions[1],  dimensions[2]]
    ulf = [ dimensions[0], -dimensions[1], -dimensions[2]]
    urb = [ dimensions[0],  dimensions[1],  dimensions[2]]
    urf = [ dimensions[0],  dimensions[1], -dimensions[2]]
    corners = [dlb, dlf, drb, drf, ulb, ulf, urb, urf]
    projected = [projection(c)[1:3] for c in corners]
    coords = ('\\coordinate (DLB) at {0};\n'
              '\\coordinate (DLF) at {1};\n'
              '\\coordinate (DRB) at {2};\n'
              '\\coordinate (DRF) at {3};\n'
              '\\coordinate (ULB) at {4};\n'
              '\\coordinate (ULF) at {5};\n'
              '\\coordinate (URB) at {6};\n'
              '\\coordinate (URF) at {7};\n').format(*[format_2d(c) for c in projected])
    draw = ('\\draw[{0}] (DLF) -- (DLB) -- (DRB);\n'
            '\\draw[{0}] (DLB) -- (ULB);\n'
            '\\draw[{0}] (DRB) '
            '-- node[below right] {{$x_3$}} (DRF) '
            '-- node[below] {{$x_2$}} (DLF) '
            '-- node[left] {{$x_1$}} (ULF) '
            '-- (ULB) -- (URB) -- cycle;').format(style)
    return coords + draw


# Generates the draw instruction for drawing a circle parallel to the equator on a sphere.
def generate_raw_latitude_2d(phi, projection, style):
    radius = cos(phi)
    offset = sin(phi)
    pr_dim = projection([offset, radius, radius])
    return '\\draw[{0}] (0, {1:0.6f}) ellipse ({2:0.6f} and {3:0.6f});'.format(
        style,
        projection([offset, 0, 0])[2],
        radius,
        radius * cos(projection.phi))


# Generates TikZ draw instructions given a list of [[phi, theta], style] elements.
def generate_raw_points_2d(projection, points, r):
    return ('\\fill[{0}] {1} circle ({2});'.format(
                point[1],
                format_2d(projection(spherical_to_cartesian(point[0]))[1:3]),
                r) for point in points)


# Generates a list of numbered colour definitions.
def generate_colours(prefix, hsbs):
    return ('\\definecolor{{{0}{1}}}{{hsb}}'
            '{{{2:0.5f}, {3:0.5f}, {4:0.5f}}}'.format(prefix, i, *hsb) for (i, hsb) in enumerate(hsbs))


# Writes table data to the file that is understood by pgfplots.
def write_table(fname, uves, max_energy):
    with open(fname, 'w') as outfile:
        for row in uves:
            for (u, v, e) in row:
                print('{0:0.5f} {1:0.5f} {2:0.5f}'.format(u, v, e / max_energy), file = outfile)
            print(file = outfile)


# Writes a table with field enery values to the file, computed via pullback.
# The two-form pulled back is determined by omega, the interspersed function is
# g_inter.
def write_field_energy(fname, omega, g_inter, window):
    x = symbols('x1 x2 x3')
    f = compose(hopf, compose(g_inter, inverse_project_stereo))
    pb = pullback(x, f, omega)
    pb_sqr = sum(pb_i ** 2 for pb_i in pb)

    eval_fns = [lambdify([x[(i + 1) % 3], x[(i + 2) % 3]],
                         pb_sqr.subs(x[i], 0)) for i in range(0, 3)]

    fields = [[[(u, v, fn(u, v)) for v in window] for u in window] for fn in eval_fns]
    max_energy = max(e for field in fields for row in field for (u, v, e) in row)

    write_table(fname + '-x2x3.dat', fields[0], max_energy)
    write_table(fname + '-x3x1.dat', fields[1], max_energy)
    write_table(fname + '-x1x2.dat', fields[2], max_energy)
