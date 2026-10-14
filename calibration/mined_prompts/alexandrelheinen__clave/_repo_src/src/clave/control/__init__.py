"""Moving the arm to the poses the tracker points at.

Five modules, split where the decisions differ. `selection` orders the
markers into a queue, `task` runs the phases of one visit, `motion` bounds
the path between two poses, and `servo` turns a commanded pose into joint
angles. Only the first three make decisions; the servo compiles a model.
`story` narrates those decisions at debug level and decides nothing.

The jaw command rides with the visit. Whether the object stays in the jaw
through the carry is what the debug run measures, and it is the open figure.
"""
