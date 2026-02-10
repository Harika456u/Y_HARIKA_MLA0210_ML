# Map Coloring Problem using CSP
# Experiment 11

# Define the map with adjacency list
australia_map = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'Q'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q': ['NT', 'SA', 'NSW'],
    'NSW': ['Q', 'SA', 'V'],
    'V': ['SA', 'NSW'],
    'T': []
}

# Available colors
colors = ['Red', 'Green', 'Blue']

# Assignment dictionary
assignment = {}

# Check if color assignment is safe
def is_safe(region, color):
    for neighbor in australia_map[region]:
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True

# Backtracking algorithm
def backtrack():
    # If all regions are assigned
    if len(assignment) == len(australia_map):
        return True

    # Select an unassigned region
    region = [r for r in australia_map if r not in assignment][0]

    # Try each color
    for color in colors:
        if is_safe(region, color):
            assignment[region] = color
            if backtrack():
                return True
            assignment.pop(region)

    return False

# Solve the problem
if backtrack():
    print("Map Coloring Solution:")
    for region in assignment:
        print(region, ":", assignment[region])
else:
    print("No solution found")
