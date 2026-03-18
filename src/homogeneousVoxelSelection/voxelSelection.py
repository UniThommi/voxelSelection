import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import numpy as np
import os

geometry_name = "currentDist"


valid_geometry_names = ["maxDist", "currentDist"]
if not geometry_name in valid_geometry_names:
    print("Invalid geometry name. It must be one of the following:")
    for name in valid_geometry_names:
        print(name)
else:
    t_zylinder = 1         # mm          
    l_voxel = 195 #* 20       # mm  sqrt((110mm)² * pi) = 194,97
    t_voxel = 1         # mm
    r_pit = 3800        # mm
    dz_pit = 1        # mm (wrong value but used in simulations)
    r_zyl_bot = 3950    # mm
    r_zyl_top = 1200    # mm
    z_offset = -5000     # mm
    h = 8900
    
    path = "/global/cfs/projectdirs/legend/users/tbuerger/sim/data/optPhotonSensitiveSurface"

    if geometry_name == "maxDist":                                        
        r_zylinder = 5858  # Apothem (innerer Radius) # 5858 in Simulation
        z_origin = 820 
        h_zylinder = h - 1
        saving_path = os.path.join(path, "maxDistZylVoxels.json")
        print("Set geometry as maxDist")
    elif geometry_name == "currentDist":
        r_zylinder = 4300  # Apothem (innerer Radius)
        z_origin = 20
        h_zylinder = h - 1
        saving_path = os.path.join(path, "currentDistZylVoxelsPMTSize.json")
        print("Set geometry as currentDist")

    # Max zylindera Werte für gleichen Index in zylinderen
    r_ref = 6000  # Apothem (innerer Radius) # 5858 in Simulation
    h_ref = 8080 # Müsste eigentlich 20 mm höher sein
    z_ref = 20

    
# --- Helper Functions ---
def is_point_in_regular_polygon(x, y, R, n=15):  # R ist innerer Kreis
    # 1. Polar-Koordinaten
    r = math.hypot(x, y)
    phi = math.atan2(y, x)

    # 2. Normiere Winkel auf [0, 2pi)
    if phi < 0:
        phi += 2 * math.pi

    # 3. Berechne Sektorindex
    theta = 2 * math.pi / n
    sector = int(phi // theta)

    # 4. Berechne Winkel zur Mitte dieses Sektors
    angle_to_side = (sector + 0.5) * theta

    # 5. Entfernung von (x, y) zur Seite (in radialer Richtung)
    angle_diff = abs(phi - angle_to_side)
    d_max = R / math.cos(angle_diff)

    return r <= d_max

def is_point_in_circle(x, y, r):
    return x**2 + y**2 <= r**2

# Berechne die kleinste Seitenlänge, die durch l_voxel teilbar ist und größer gleich dem Durchmesser ist
diameter = 2 * r_pit
grid_size = math.ceil(diameter / l_voxel) * l_voxel
half_grid = grid_size / 2

x_start = -half_grid
y_start = -half_grid
z_bottom = 20 + z_offset
z_top = z_bottom + dz_pit
z_mid = (z_bottom + z_top) / 2

# Anzahl Schritte pro Richtung
steps = int(grid_size / l_voxel)

# List of valid voxels
voxels_pit = []

# Für 3d Plot
filled = np.zeros((steps, steps, 1), dtype=bool)

fig, ax = plt.subplots(figsize=(10, 10))

circle = patches.Circle((0,0), r_pit, linewidth=1.0, edgecolor='black')
ax.add_patch(circle)

z = 0

for y_idx in range(steps):
    for x_idx in range(steps):
        x = x_start + x_idx * l_voxel 
        y = y_start + y_idx * l_voxel

        # Ecken des kleinen Quadrats
        corners = [
            (x, y),
            (x + l_voxel, y),
            (x, y + l_voxel),
            (x + l_voxel, y + l_voxel),
        ]

        # Wenn eine Ecke im Kreis liegt → Quadrat behalten
        if any(cx**2 + cy**2 <= r_pit**2 for cx, cy in corners):
            mid_x = x + l_voxel / 2
            mid_y = y + l_voxel / 2

            # Index im Format zzyyxx (mit führenden Nullen, 2 Stellen je Komponente)
            index_str = f"{z:02d}{y_idx:02d}{x_idx:02d}"
            
            # Berechne alle 8 Eckpunkte des Voxels
            z_min = z_mid - dz_pit / 2
            z_max = z_mid + dz_pit / 2
            corners_3d = [
                (mid_x - l_voxel/2, mid_y - l_voxel/2, z_min),  # unten links vorne
                (mid_x + l_voxel/2, mid_y - l_voxel/2, z_min),  # unten rechts vorne
                (mid_x + l_voxel/2, mid_y + l_voxel/2, z_min),  # unten rechts hinten
                (mid_x - l_voxel/2, mid_y + l_voxel/2, z_min),  # unten links hinten
                (mid_x - l_voxel/2, mid_y - l_voxel/2, z_max),  # oben links vorne
                (mid_x + l_voxel/2, mid_y - l_voxel/2, z_max),  # oben rechts vorne
                (mid_x + l_voxel/2, mid_y + l_voxel/2, z_max),  # oben rechts hinten
                (mid_x - l_voxel/2, mid_y + l_voxel/2, z_max)   # oben links hinten
            ]

            voxels_pit.append({
                "index": index_str,
                "center": (mid_x, mid_y, z_mid),
                "corners": corners_3d,  # Alle 8 Eckpunkte
                "layer": "pit",
            })

            rect = patches.Rectangle((mid_x - l_voxel/2, mid_y - l_voxel/2), l_voxel, l_voxel,
                                     linewidth = 0.2, edgecolor="blue", facecolor="none")
            ax.add_patch(rect)
            ax.text(mid_x, mid_y, index_str, fontsize=7, ha="center", va="center")

            # Mark as filled in 3D grid
            filled[x_idx, y_idx, 0] = True

# 2d Plot
ax.set_title("Voxel Gitter im Zylinder (XY-Schnitt)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-r_pit*1.1, r_pit*1.1)
ax.set_ylim(-r_pit*1.1, r_pit*1.1)
plt.grid(True)
plt.show()


# 3d Plot
# Create 3D plot using voxel positions
fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')

# Plot each voxel using its stored corners
for voxel in voxels_pit:
    corners = voxel["corners"]
    
    # Define faces using corner indices
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom face
        [corners[4], corners[5], corners[6], corners[7]],  # top face
        [corners[0], corners[1], corners[5], corners[4]],  # front face
        [corners[2], corners[3], corners[7], corners[6]],  # back face
        [corners[0], corners[3], corners[7], corners[4]],  # left face
        [corners[1], corners[2], corners[6], corners[5]]   # right face
    ]
    
    # Plot each face
    for face in faces:
        ax3d.add_collection3d(Poly3DCollection(
            [face], alpha=0.3, linewidths=0.5, edgecolor='k', facecolor='cyan'
        ))

# Set plot properties
ax3d.set_title("3D Voxel Representation")
ax3d.set_xlabel("x [mm]")
ax3d.set_ylabel("y [mm]")
ax3d.set_zlabel("z [mm]")
ax3d.set_xlim(-r_pit*1.1, r_pit*1.1)
ax3d.set_ylim(-r_pit*1.1, r_pit*1.1)
ax3d.set_zlim(z_offset, z_offset + dz_pit)

# Set equal aspect ratio
ax3d.set_box_aspect([2*r_pit, 2*r_pit, dz_pit])

plt.tight_layout()
plt.show()

diameter = 2 * r_ref

grid_size = math.ceil(diameter / l_voxel) * l_voxel
half_grid = grid_size / 2
x_start = -half_grid
y_start = -half_grid
steps = int(grid_size / l_voxel)

if geometry_name == "minDist":
    voxels_bot = []
    fig2d, ax2d = plt.subplots(figsize=(10, 10))
    circle = patches.Circle((0, 0), r_zyl_bot, linewidth=1.0, 
                            edgecolor='black', facecolor='blue', alpha=0.3)
    ax2d.add_patch(circle)

    circle = patches.Circle((0, 0), r_zylinder, linewidth=1.0, 
                            edgecolor='black', facecolor='red', alpha=0.3)
    ax2d.add_patch(circle)



else:
    # Z-coordinates (single layer)
    z_bottom = z_origin + z_offset
    z_top = z_bottom + t_zylinder
    z_mid = (z_bottom + z_top) / 2

    # --- Voxel Generation ---
    voxels_bot = []

    fig2d, ax2d = plt.subplots(figsize=(10, 10))

    # Draw reference shapes
    circle = patches.Circle((0, 0), r_zyl_bot, linewidth=1.0, 
                            edgecolor='black', facecolor='blue', alpha=0.3)
    ax2d.add_patch(circle)

    circle = patches.Circle((0, 0), r_zylinder, linewidth=1.0, 
                            edgecolor='black', facecolor='red', alpha=0.3)
    ax2d.add_patch(circle)

    z = 1

    # Generate voxels
    for y_idx in range(steps):
        for x_idx in range(steps):
            x = x_start + x_idx * l_voxel 
            y = y_start + y_idx * l_voxel
            
            corners_2d = [
                (x, y),
                (x + l_voxel, y),
                (x, y + l_voxel),
                (x + l_voxel, y + l_voxel)
            ]
            
            # Check if voxel should be included
            if any(is_point_in_circle(cx, cy, r_zylinder) and not is_point_in_circle(cx, cy, r_zyl_bot) for cx, cy in corners_2d):
                mid_x = x + l_voxel / 2
                mid_y = y + l_voxel / 2

                index_str = f"{z:02d}{y_idx:02d}{x_idx:02d}"
                
                # Calculate 3D corners for the voxel
                x_min = mid_x - l_voxel / 2
                x_max = mid_x + l_voxel / 2
                y_min = mid_y - l_voxel / 2
                y_max = mid_y + l_voxel / 2
                z_min = z_mid - t_zylinder / 2
                z_max = z_mid + t_zylinder / 2
                
                corners_3d = [
                    (x_min, y_min, z_min),
                    (x_max, y_min, z_min),
                    (x_min, y_max, z_min),
                    (x_max, y_max, z_min),
                    (x_min, y_min, z_max),
                    (x_max, y_min, z_max),
                    (x_min, y_max, z_max),
                    (x_max, y_max, z_max)
                ]
                
                # Voxel metadata with corners instead of rotation/size
                voxels_bot.append({
                    "index": index_str,
                    "center": (mid_x, mid_y, z_mid),
                    "layer": "bot",
                    "corners": corners_3d
                })
                
                # 2D visualization
                rect = patches.Rectangle((mid_x - l_voxel/2, mid_y - l_voxel/2), l_voxel, l_voxel,
                                        linewidth = 0.2, edgecolor="blue", facecolor="none")
                ax2d.add_patch(rect)
                ax2d.text(mid_x, mid_y, index_str, fontsize=6, ha="center", va="center")
                

# --- 2D Plot Settings ---
ax2d.set_title("Voxel Grid zylinder Bot (XY Plane)")
ax2d.set_xlabel("X [mm]")
ax2d.set_ylabel("Y [mm]")
ax2d.set_aspect('equal')
ax2d.set_xlim(-r_zylinder, r_zylinder)
ax2d.set_ylim(-r_zylinder, r_zylinder)
ax2d.grid(True)

# 3d Plot
# Create 3D plot using voxel positions
fig3d = plt.figure(figsize=(12, 10))
ax3d = fig3d.add_subplot(111, projection='3d')

# Prepare arrays for voxel positions
voxel_positions = np.array([voxel["center"] for voxel in voxels_bot])  # Fixed variable name

# Create voxel visualization from corners
for voxel in voxels_bot:
    corners = voxel["corners"]
    
    # Define faces using the corner indices (consistent ordering)
    faces = [
        [corners[0], corners[1], corners[3], corners[2]],  # bottom
        [corners[4], corners[5], corners[7], corners[6]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[0], corners[2], corners[6], corners[4]],  # left
        [corners[1], corners[3], corners[7], corners[5]]   # right
    ]
    
    # Plot each face
    for face in faces:
        ax3d.add_collection3d(Poly3DCollection(
            [face], alpha=0.3, linewidths=0.5, edgecolor='k', facecolor='cyan'
        ))

# Set plot properties
ax3d.set_title("3D Voxel zylinder Bot Representation")
ax3d.set_xlabel("x [mm]")
ax3d.set_ylabel("y [mm]")
ax3d.set_zlabel("z [mm]")
ax3d.set_xlim(-r_zylinder*1.1, r_zylinder*1.1)
ax3d.set_ylim(-r_zylinder*1.1, r_zylinder*1.1)
ax3d.set_zlim(z_bottom - 10, z_bottom + 10)
# Set equal aspect ratio
ax3d.set_box_aspect([2*r_zylinder, 2*r_zylinder, dz_pit])

plt.tight_layout()
plt.show()

# Grid setup
diameter = 2 * r_ref

grid_size = math.ceil(diameter / l_voxel) * l_voxel
half_grid = grid_size / 2
x_start = -half_grid
y_start = -half_grid
steps = int(grid_size / l_voxel)

# Z-coordinates (single layer)
z_bottom = z_origin + h_zylinder + z_offset
z_top = z_bottom + t_zylinder
z_mid = (z_bottom + z_top) / 2

# --- Voxel Generation ---
voxels_top = []

# Identity rotation matrix is no longer stored
fig2d, ax2d = plt.subplots(figsize=(10, 10))

# Draw reference shapes
circle = patches.Circle((0, 0), r_zyl_top, linewidth=1.0, 
                        edgecolor='black', facecolor='blue', alpha=0.3)
ax2d.add_patch(circle)

circle = patches.Circle((0, 0), r_zylinder, linewidth=1.0, 
                        edgecolor='black', facecolor='red', alpha=0.3)
ax2d.add_patch(circle)



z = 99

# Generate voxels
for y_idx in range(steps):
    for x_idx in range(steps):
        x = x_start + x_idx * l_voxel 
        y = y_start + y_idx * l_voxel
        
        corners = [
            (x, y),
            (x + l_voxel, y),
            (x, y + l_voxel),
            (x + l_voxel, y + l_voxel)
        ]
        
        # Check if voxel should be included
        if any(is_point_in_circle(cx, cy, r_zylinder) and not is_point_in_circle(cx, cy, r_zyl_top) for cx, cy in corners):
            mid_x = x + l_voxel / 2
            mid_y = y + l_voxel / 2
            # Index im Format zzyyxx (mit führenden Nullen, 2 Stellen je Komponente)
            index_str = f"{z:02d}{y_idx:02d}{x_idx:02d}"

            # Calculate all 8 corners of the voxel
            half_x = l_voxel / 2
            half_y = l_voxel / 2
            half_z = t_zylinder / 2
            corners3d = [
                (mid_x - half_x, mid_y - half_y, z_mid - half_z),  # bottom-front-left
                (mid_x + half_x, mid_y - half_y, z_mid - half_z),  # bottom-front-right
                (mid_x + half_x, mid_y + half_y, z_mid - half_z),  # bottom-back-right
                (mid_x - half_x, mid_y + half_y, z_mid - half_z),  # bottom-back-left
                (mid_x - half_x, mid_y - half_y, z_mid + half_z),  # top-front-left
                (mid_x + half_x, mid_y - half_y, z_mid + half_z),  # top-front-right
                (mid_x + half_x, mid_y + half_y, z_mid + half_z),  # top-back-right
                (mid_x - half_x, mid_y + half_y, z_mid + half_z)   # top-back-left
            ]
            
            voxels_top.append({
                "index": index_str,
                "center": (mid_x, mid_y, z_mid),
                "corners": corners3d,  # store all 8 corners
                "layer": "top",
            })
            
            # 2D visualization remains unchanged
            rect = patches.Rectangle((mid_x - l_voxel/2, mid_y - l_voxel/2), l_voxel, l_voxel,
                                     linewidth = 0.2, edgecolor="blue", facecolor="none")
            ax2d.add_patch(rect)
            ax2d.text(mid_x, mid_y, index_str, fontsize=6, ha="center", va="center")
            

# --- 2D Plot Settings ---
ax2d.set_title("Voxel Grid zylinder Top (XY Plane)")
ax2d.set_xlabel("X [mm]")
ax2d.set_ylabel("Y [mm]")
ax2d.set_aspect('equal')
ax2d.set_xlim(-r_zylinder*1.1, r_zylinder*1.1)
ax2d.set_ylim(-r_zylinder*1.1,r_zylinder*1.1)
ax2d.grid(True)

# Create 3D plot using voxel positions
fig3d = plt.figure(figsize=(12, 10))
ax3d = fig3d.add_subplot(111, projection='3d')

# Prepare arrays for voxel positions
voxel_positions = np.array([voxel["center"] for voxel in voxels_top])

# Create voxel edges using stored corners
for voxel in voxels_top:
    corners = np.array(voxel["corners"])
    
    # Define the faces of the voxel using corner indices
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[0], corners[3], corners[7], corners[4]],  # left
        [corners[1], corners[2], corners[6], corners[5]]   # right
    ]
    
    # Plot each face
    for face in faces:
        ax3d.add_collection3d(Poly3DCollection(
            [face], alpha=0.3, linewidths=0.5, edgecolor='k', facecolor='cyan'
        ))

# Set plot properties
ax3d.set_title("3D Voxel zylinder Top Representation")
ax3d.set_xlabel("x [mm]")
ax3d.set_ylabel("y [mm]")
ax3d.set_zlabel("z [mm]")
ax3d.set_xlim(-r_zylinder*1.1, r_zylinder*1.1)
ax3d.set_ylim(-r_zylinder*1.1, r_zylinder*1.1)
ax3d.set_zlim(z_bottom - 10, z_bottom + 10)

# Set equal aspect ratio
ax3d.set_box_aspect([2*r_zylinder, 2*r_zylinder, t_zylinder])

plt.tight_layout()
plt.show()

n_theta = int(round(2 * np.pi * r_zylinder / l_voxel))
angle_per_segment = 2 * np.pi / n_theta

voxels_wall = []
w_index = 30

# Globale Z-Indizes (sicher gegen negative Werte)
z_base_global = z_origin + z_offset
start_zz_global = 0
end_zz_global = int(np.ceil((z_base_global + h_zylinder - (z_ref + z_offset)) / l_voxel)) 


for i in range(n_theta):
    theta1 = i * angle_per_segment
    theta2 = (i + 1) * angle_per_segment
    
    for zz_global in range(start_zz_global, end_zz_global):
        # Globale Positionierung (immer relativ zu z_ref)
        z_bottom_global = z_ref + zz_global * l_voxel + z_offset
        z_top_global = z_bottom_global + l_voxel
        
        # Prüfe Überlappung mit Zylinder
        z_overlap_min = max(z_bottom_global, z_base_global)
        z_overlap_max = min(z_top_global, z_base_global + h_zylinder)
        if z_overlap_min >= z_overlap_max:  # Kein Schnitt -> überspringen
            continue
        
        # Innere Punkte (Radius = r_zylinder)
        bl_inner = np.array([r_zylinder * np.cos(theta1), r_zylinder * np.sin(theta1), z_bottom_global])
        br_inner = np.array([r_zylinder * np.cos(theta2), r_zylinder * np.sin(theta2), z_bottom_global])
        tl_inner = np.array([r_zylinder * np.cos(theta1), r_zylinder * np.sin(theta1), z_top_global])
        tr_inner = np.array([r_zylinder * np.cos(theta2), r_zylinder * np.sin(theta2), z_top_global])
        
        # Äußere Punkte (Radius = r_zylinder + t_zylinder)
        bl_outer = np.array([(r_zylinder + t_zylinder) * np.cos(theta1), (r_zylinder + t_zylinder) * np.sin(theta1), z_bottom_global])
        br_outer = np.array([(r_zylinder + t_zylinder) * np.cos(theta2), (r_zylinder + t_zylinder) * np.sin(theta2), z_bottom_global])
        tl_outer = np.array([(r_zylinder + t_zylinder) * np.cos(theta1), (r_zylinder + t_zylinder) * np.sin(theta1), z_top_global])
        tr_outer = np.array([(r_zylinder + t_zylinder) * np.cos(theta2), (r_zylinder + t_zylinder) * np.sin(theta2), z_top_global])
        
        # Kombiniere alle Ecken
        all_corners = [
            bl_inner.tolist(), br_inner.tolist(), 
            tr_inner.tolist(), tl_inner.tolist(),
            bl_outer.tolist(), br_outer.tolist(),
            tr_outer.tolist(), tl_outer.tolist()
        ]
        
        # Mittelpunkt als Durchschnitt der Ecken
        center = np.mean(all_corners, axis=0).tolist()
        
        # Erstelle Voxel mit eindeutigem Index
        index = f"{w_index:02d}{zz_global:02d}{i:02d}"
        voxels_wall.append({
            "index": index,
            "center": center,
            "corners": all_corners,
            "layer": "wall",
        })

# Plot the voxels using the new corner-based representation
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for voxel in voxels_wall:
    corners = voxel['corners']  # Get the precomputed corners
    # Extract all 8 corners in order
    bl_inner = corners[0]
    br_inner = corners[1]
    tr_inner = corners[2]
    tl_inner = corners[3]
    bl_outer = corners[4]
    br_outer = corners[5]
    tr_outer = corners[6]
    tl_outer = corners[7]
    
    # Define the 6 faces using the corner points
    faces = [
        [bl_inner, br_inner, br_outer, bl_outer],  # bottom face
        [tl_inner, tr_inner, tr_outer, tl_outer],  # top face
        [bl_inner, tl_inner, tl_outer, bl_outer],  # left face (along wall)
        [br_inner, tr_inner, tr_outer, br_outer],  # right face (along wall)
        [bl_inner, br_inner, tr_inner, tl_inner],  # inner face (toward prism center)
        [bl_outer, br_outer, tr_outer, tl_outer]   # outer face
    ]
    
    # Add voxel to plot
    ax.add_collection3d(Poly3DCollection(
        faces, 
        facecolors='cyan', 
        edgecolors='k', 
        linewidths=0.5, 
        alpha=0.5
    ))

# Configure plot
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title(f'Voxelized Zylinder Walls')
ax.set_xlim(-r_zylinder*1.2, r_zylinder*1.2)
ax.set_ylim(-r_zylinder*1.2, r_zylinder*1.2)
ax.set_zlim(z_offset, z_origin + h_zylinder + z_offset + 10)

plt.tight_layout()
plt.show()

# Save all Voxels
all_voxels = np.concatenate([voxels_pit, voxels_bot, voxels_top, voxels_wall])

print("Anzahl Pit Voxel: ", len(voxels_pit))
print("Anzahl Zylinder Bot Voxel: ", len(voxels_bot))
print("Anzahl Zylinderder Top Voxel: ", len(voxels_top))
print("Anzahl Zylinderderder Wall Voxel: ", len(voxels_wall))
print("Anzahl aller Voxel: ", len(all_voxels))

with open(saving_path, "w") as f:
    json.dump(all_voxels.tolist(), f, indent=2)


pmt_radius = 131  # mm
total_selection = 300

all_voxels = np.concatenate([voxels_pit, voxels_bot, voxels_top, voxels_wall])

# Helper-Funktion: Prüft ob PMT im gültigen Bereich liegt
def is_valid_pmt_position(center, layer, pmt_r=pmt_radius):
    """Prüft ob ein PMT mit gegebenem Radius an der Position platziert werden kann"""
    x, y, z = center
    r_center = np.sqrt(x**2 + y**2)

    if layer == "pit":
        return r_center + pmt_r <= r_pit
    elif layer == "bot":
        return (r_center - pmt_r >= r_zyl_bot) and (r_center + pmt_r <= r_zylinder)
    elif layer == "top":
        return (r_center - pmt_r >= r_zyl_top) and (r_center + pmt_r <= r_zylinder)
    elif layer == "wall":
        z_min_allowed = z_base_global + pmt_r
        z_max_allowed = z_base_global + h_zylinder - pmt_r
        return z_min_allowed <= z <= z_max_allowed
    return False


#: Fibonacci-Spiral Funktionen
def fibonacci_disk(n_points, r_inner, r_outer):
    """Erzeugt gleichmäßig verteilte Punkte in einem Ring (Disk mit innerem Loch)"""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)
    theta = 2 * np.pi * indices / golden_ratio
    r = np.sqrt(r_inner**2 + (r_outer**2 - r_inner**2) * (indices + 0.5) / n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def fibonacci_cylinder_wall(n_points, radius, z_min, z_max):
    """Erzeugt gleichmäßig verteilte Punkte auf Zylinderwand"""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)
    theta = 2 * np.pi * indices / golden_ratio
    z = z_min + (z_max - z_min) * (indices + 0.5) / n_points
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def find_nearest_voxel(point, voxel_list):
    """Findet den nächsten Voxel zu einem gegebenen Punkt"""
    if len(voxel_list) == 0:
        return None
    centers = np.array([v["center"] for v in voxel_list])
    distances = np.linalg.norm(centers - point, axis=1)
    nearest_idx = np.argmin(distances)
    return voxel_list[nearest_idx]


# =============================================================================
# Flächenproportionale Verteilung statt Voxel-proportional
# =============================================================================

area_pit = np.pi * r_pit**2
area_bot = np.pi * (r_zylinder**2 - r_zyl_bot**2)
area_top = np.pi * (r_zylinder**2 - r_zyl_top**2)
area_wall = 2 * np.pi * r_zylinder * h_zylinder
total_area = area_pit + area_bot + area_top + area_wall

target_density = total_selection / total_area

# Filtere gültige Voxel für jede Schicht
valid_pit = [v for v in voxels_pit if is_valid_pmt_position(v["center"], v["layer"])]
valid_bot = [v for v in voxels_bot if is_valid_pmt_position(v["center"], v["layer"])]
valid_top = [v for v in voxels_top if is_valid_pmt_position(v["center"], v["layer"])]
valid_wall = [v for v in voxels_wall if is_valid_pmt_position(v["center"], v["layer"])]

print("\n=== Gültige Voxel nach PMT-Randbedingungen ===")
print(f"Gültige Pit Voxel: {len(valid_pit)} von {len(voxels_pit)}")
print(f"Gültige Bot Voxel: {len(valid_bot)} von {len(voxels_bot)}")
print(f"Gültige Top Voxel: {len(valid_top)} von {len(voxels_top)}")
print(f"Gültige Wall Voxel: {len(valid_wall)} von {len(voxels_wall)}")

# Flächenproportionale Anzahl
n_pit = int(np.round(target_density * area_pit))
n_bot = int(np.round(target_density * area_bot))
n_top = int(np.round(target_density * area_top))
n_wall = int(np.round(target_density * area_wall))

# Korrektur falls Rundung nicht exakt 300 ergibt
diff = total_selection - (n_pit + n_bot + n_top + n_wall)
if diff != 0:
    counts = [n_pit, n_bot, n_top, n_wall]
    max_idx = counts.index(max(counts))
    if max_idx == 0: n_pit += diff
    elif max_idx == 1: n_bot += diff
    elif max_idx == 2: n_top += diff
    else: n_wall += diff

# Bot-Check: genug Positionen (valide + invalide)?
print(f"\n=== Bot-Verfügbarkeitscheck ===")
print(f"Bot benötigt: {n_bot}, valide Positionen: {len(valid_bot)}, "
      f"alle Positionen: {len(voxels_bot)}")

bot_use_all_voxels = False
if len(valid_bot) >= n_bot:
    print(f"✅ Bot: genug valide Positionen")
elif len(voxels_bot) >= n_bot:
    bot_use_all_voxels = True
    print(f"⚠️  Bot: nur {len(valid_bot)} valide, nutze alle {len(voxels_bot)} Voxel "
          f"(inkl. invalide, müssen manuell verschoben werden)")
else:
    raise RuntimeError(
        f"❌ ABBRUCH: Bot benötigt {n_bot} PMTs, aber nur {len(voxels_bot)} "
        f"Voxel vorhanden (davon {len(valid_bot)} valide). "
        f"Nicht genug Positionen, selbst mit invaliden Voxeln.")

# Dichtevergleich
print(f"\n=== Selektion (flächenproportional) ===")
print(f"Zieldichte: {target_density:.6e} PMTs/mm²")
print(f"Zu selektierende Pit Voxel: {n_pit}")
print(f"Zu selektierende Bot Voxel: {n_bot}")
print(f"Zu selektierende Top Voxel: {n_top}")
print(f"Zu selektierende Wall Voxel: {n_wall}")
print(f"Gesamt: {n_pit + n_bot + n_top + n_wall}")

print(f"\n  {'Area':<6} {'N_PMTs':>7} {'Fläche (M mm²)':>15} {'Dichte':>14} {'Abw. von Ziel':>14}")
print(f"  " + "-" * 58)
for name, n, a in [('pit', n_pit, area_pit), ('bot', n_bot, area_bot),
                    ('top', n_top, area_top), ('wall', n_wall, area_wall)]:
    d = n / a
    dev = (d - target_density) / target_density * 100
    print(f"  {name:<6} {n:>7} {a/1e6:>15.2f} {d:>14.6e} {dev:>+13.1f}%")

# =============================================================================
# Fibonacci-Spiralen basierte Selektion
# =============================================================================

selected_pit = []
selected_bot = []
selected_top = []
selected_wall = []

# Pit: Fibonacci Disk
if len(valid_pit) > 0 and n_pit > 0:
    print(f"\nGeneriere {n_pit} Fibonacci-Punkte für Pit...")
    fib_points_2d = fibonacci_disk(n_pit, 0, r_pit - pmt_radius)
    z_pit = voxels_pit[0]["center"][2]
    fib_points = np.column_stack([fib_points_2d, np.full(n_pit, z_pit)])

    used_indices = set()
    for point in fib_points:
        min_dist = float('inf')
        nearest = None
        nearest_idx = None
        for idx, voxel in enumerate(valid_pit):
            if idx in used_indices:
                continue
            dist = np.linalg.norm(np.array(voxel["center"]) - point)
            if dist < min_dist:
                min_dist = dist
                nearest = voxel
                nearest_idx = idx
        if nearest is not None:
            selected_pit.append(nearest)
            used_indices.add(nearest_idx)

# Bot: Fibonacci Disk (mit Fallback auf alle Voxel)
if n_bot > 0:
    print(f"Generiere {n_bot} Fibonacci-Punkte für Bot...")
    fib_points_2d = fibonacci_disk(n_bot, r_zyl_bot + pmt_radius, r_zylinder - pmt_radius)
    z_bot = voxels_bot[0]["center"][2]
    fib_points = np.column_stack([fib_points_2d, np.full(n_bot, z_bot)])

    bot_candidates = list(voxels_bot) if bot_use_all_voxels else valid_bot

    used_indices = set()
    for point in fib_points:
        min_dist = float('inf')
        nearest = None
        nearest_idx = None
        for idx, voxel in enumerate(bot_candidates):
            if idx in used_indices:
                continue
            dist = np.linalg.norm(np.array(voxel["center"]) - point)
            if dist < min_dist:
                min_dist = dist
                nearest = voxel
                nearest_idx = idx
        if nearest is not None:
            selected_bot.append(nearest)
            used_indices.add(nearest_idx)

    if bot_use_all_voxels:
        n_invalid = sum(1 for v in selected_bot
                        if not is_valid_pmt_position(v["center"], v["layer"]))
        if n_invalid > 0:
            print(f"  ⚠️  {n_invalid} Bot-PMTs auf invaliden Positionen "
                  f"(müssen in Simulation verschoben werden)")

# Top: Fibonacci Ring
if len(valid_top) > 0 and n_top > 0:
    print(f"Generiere {n_top} Fibonacci-Punkte für Top...")
    fib_points_2d = fibonacci_disk(n_top, r_zyl_top + pmt_radius, r_zylinder - pmt_radius)
    z_top = voxels_top[0]["center"][2]
    fib_points = np.column_stack([fib_points_2d, np.full(n_top, z_top)])

    used_indices = set()
    for point in fib_points:
        min_dist = float('inf')
        nearest = None
        nearest_idx = None
        for idx, voxel in enumerate(valid_top):
            if idx in used_indices:
                continue
            dist = np.linalg.norm(np.array(voxel["center"]) - point)
            if dist < min_dist:
                min_dist = dist
                nearest = voxel
                nearest_idx = idx
        if nearest is not None:
            selected_top.append(nearest)
            used_indices.add(nearest_idx)

# Wall: Fibonacci Zylinderwand
if len(valid_wall) > 0 and n_wall > 0:
    print(f"Generiere {n_wall} Fibonacci-Punkte für Wall...")
    z_min_wall = z_base_global + pmt_radius
    z_max_wall = z_base_global + h_zylinder - pmt_radius
    fib_points = fibonacci_cylinder_wall(n_wall, r_zylinder, z_min_wall, z_max_wall)

    used_indices = set()
    for point in fib_points:
        min_dist = float('inf')
        nearest = None
        nearest_idx = None
        for idx, voxel in enumerate(valid_wall):
            if idx in used_indices:
                continue
            dist = np.linalg.norm(np.array(voxel["center"]) - point)
            if dist < min_dist:
                min_dist = dist
                nearest = voxel
                nearest_idx = idx
        if nearest is not None:
            selected_wall.append(nearest)
            used_indices.add(nearest_idx)

# Kombiniere alle selektierten Voxel
selected_voxels = selected_pit + selected_bot + selected_top + selected_wall

print(f"\n=== Tatsächlich selektiert ===")
print(f"Selektierte Pit Voxel: {len(selected_pit)}")
print(f"Selektierte Bot Voxel: {len(selected_bot)}")
print(f"Selektierte Top Voxel: {len(selected_top)}")
print(f"Selektierte Wall Voxel: {len(selected_wall)}")
print(f"Gesamt selektiert: {len(selected_voxels)}")


# =============================================================================
# Homogenitäts-Check: Nearest-Neighbor Distance
# =============================================================================

from scipy.spatial import KDTree

def compute_nn_stats(voxels):
    """Berechne Nearest-Neighbor-Distanz Statistiken."""
    if len(voxels) < 2:
        return np.array([]), {}
    centers = np.array([v["center"] for v in voxels])
    tree = KDTree(centers)
    dists, _ = tree.query(centers, k=2)
    nn_dists = dists[:, 1]
    stats = {
        'mean': np.mean(nn_dists),
        'std': np.std(nn_dists),
        'min': np.min(nn_dists),
        'max': np.max(nn_dists),
        'cv': np.std(nn_dists) / np.mean(nn_dists) if np.mean(nn_dists) > 0 else 0,
    }
    return nn_dists, stats


def compute_nn_stats_wall(voxels, r_cyl):
    """NN-Distanz auf Zylinderoberfläche (geodätisch: Bogenlänge in φ, euklidisch in z)."""
    if len(voxels) < 2:
        return np.array([]), {}
    centers = np.array([v["center"] for v in voxels])
    phi = np.arctan2(centers[:, 1], centers[:, 0])
    z = centers[:, 2]
    n = len(centers)
    nn_dists = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dphi = np.abs(phi[i] - phi[j])
            dphi = min(dphi, 2 * np.pi - dphi)
            arc = r_cyl * dphi
            dz = z[i] - z[j]
            dist = np.sqrt(arc**2 + dz**2)
            nn_dists[i] = min(nn_dists[i], dist)
    stats = {
        'mean': np.mean(nn_dists),
        'std': np.std(nn_dists),
        'min': np.min(nn_dists),
        'max': np.max(nn_dists),
        'cv': np.std(nn_dists) / np.mean(nn_dists) if np.mean(nn_dists) > 0 else 0,
    }
    return nn_dists, stats


print("\n" + "=" * 70)
print("HOMOGENITÄTS-CHECK: Nearest-Neighbor-Distanz Analyse")
print("=" * 70)
print(f"\n  {'Area':<6} {'N':>4} {'Mean NN':>10} {'Std NN':>10} {'Min NN':>10} "
      f"{'Max NN':>10} {'CV':>8}")
print(f"  " + "-" * 60)

nn_data = {}
for name, voxels in [('pit', selected_pit), ('bot', selected_bot),
                      ('top', selected_top), ('wall', selected_wall)]:
    if name == 'wall':
        nn_dists, stats = compute_nn_stats_wall(voxels, r_zylinder)
    else:
        nn_dists, stats = compute_nn_stats(voxels)
    nn_data[name] = (nn_dists, stats)
    if stats:
        print(f"  {name:<6} {len(voxels):>4} {stats['mean']:>10.1f} {stats['std']:>10.1f} "
              f"{stats['min']:>10.1f} {stats['max']:>10.1f} {stats['cv']:>8.3f}")
    else:
        print(f"  {name:<6} {len(voxels):>4}   (zu wenig Punkte)")

print(f"\n  CV = Variationskoeffizient (std/mean). Kleiner = homogener.")
print(f"  Fibonacci-Gitter: CV ≈ 0.05-0.15, nach Voxel-Snapping: CV ≈ 0.10-0.25 akzeptabel")


# =============================================================================
# Visualisierung: 3D Plot (selektierte vs. alle Voxel)
# =============================================================================

fig_sel = plt.figure(figsize=(14, 10))
ax_sel = fig_sel.add_subplot(111, projection='3d')

# Plotte alle Voxel transparent
for voxel in all_voxels:
    corners = voxel["corners"]
    if voxel["layer"] == "wall":
        bl_inner, br_inner, tr_inner, tl_inner = corners[0:4]
        bl_outer, br_outer, tr_outer, tl_outer = corners[4:8]
        faces = [
            [bl_inner, br_inner, br_outer, bl_outer],
            [tl_inner, tr_inner, tr_outer, tl_outer],
            [bl_inner, tl_inner, tl_outer, bl_outer],
            [br_inner, tr_inner, tr_outer, br_outer],
            [bl_inner, br_inner, tr_inner, tl_inner],
            [bl_outer, br_outer, tr_outer, tl_outer]
        ]
    else:
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[0], corners[3], corners[7], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]]
        ]

    ax_sel.add_collection3d(Poly3DCollection(
        faces, facecolors='lightgray', edgecolors='gray',
        linewidths=0.2, alpha=0.1
    ))

# Plotte selektierte Voxel hervorgehoben
for voxel in selected_voxels:
    corners = voxel["corners"]
    center = voxel["center"]

    if voxel["layer"] == "pit":
        color = 'red'
    elif voxel["layer"] == "bot":
        color = 'blue'
    elif voxel["layer"] == "top":
        color = 'green'
    else:
        color = 'orange'

    if voxel["layer"] == "wall":
        bl_inner, br_inner, tr_inner, tl_inner = corners[0:4]
        bl_outer, br_outer, tr_outer, tl_outer = corners[4:8]
        faces = [
            [bl_inner, br_inner, br_outer, bl_outer],
            [tl_inner, tr_inner, tr_outer, tl_outer],
            [bl_inner, tl_inner, tl_outer, bl_outer],
            [br_inner, tr_inner, tr_outer, br_outer],
            [bl_inner, br_inner, tr_inner, tl_inner],
            [bl_outer, br_outer, tr_outer, tl_outer]
        ]
    else:
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[0], corners[3], corners[7], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]]
        ]

    ax_sel.add_collection3d(Poly3DCollection(
        faces, facecolors=color, edgecolors='black',
        linewidths=0.8, alpha=0.7
    ))

    ax_sel.scatter(center[0], center[1], center[2],
                   color=color, s=50, marker='o', edgecolors='black')

ax_sel.set_xlabel('X (mm)')
ax_sel.set_ylabel('Y (mm)')
ax_sel.set_zlabel('Z (mm)')
ax_sel.set_title(f'Selektierte PMT-Positionen (n={len(selected_voxels)}, flächenproportional)\n'
                 f'Rot=Pit, Blau=Bot, Grün=Top, Orange=Wall')
ax_sel.set_xlim(-r_zylinder*1.1, r_zylinder*1.1)
ax_sel.set_ylim(-r_zylinder*1.1, r_zylinder*1.1)
ax_sel.set_zlim(z_offset, z_origin + h_zylinder + z_offset + 100)

plt.tight_layout()
plt.savefig(os.path.join(path, "pmt_selection_3d.png"), dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Homogenitäts-Check: 2D Positionen + NN-Statistik (separate Figure)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for idx, (name, voxels, r_min_plot, r_max_plot) in enumerate([
    ('Pit', selected_pit, 0, r_pit),
    ('Bot', selected_bot, r_zyl_bot, r_zylinder),
    ('Top', selected_top, r_zyl_top, r_zylinder),
    ('Wall', selected_wall, None, None),
]):
    ax = axes[idx // 2, idx % 2]
    if name.lower() == 'wall':
        if len(voxels) > 0:
            centers = np.array([v["center"] for v in voxels])
            phi = np.arctan2(centers[:, 1], centers[:, 0])
            z_vals = centers[:, 2]
            ax.scatter(phi, z_vals, s=15, c='blue', alpha=0.7)
            for p, zv in zip(phi, z_vals):
                circle = plt.Circle((p, zv), pmt_radius / r_zylinder,
                                    fill=False, edgecolor='blue',
                                    linewidth=0.3, alpha=0.5)
                ax.add_patch(circle)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(z_base_global, z_base_global + h_zylinder)
        ax.set_xlabel('φ [rad]')
        ax.set_ylabel('z [mm]')
    else:
        if len(voxels) > 0:
            centers = np.array([v["center"] for v in voxels])
            ax.scatter(centers[:, 0], centers[:, 1], s=15, c='blue', alpha=0.7)
            for c in centers:
                circle = plt.Circle((c[0], c[1]), pmt_radius,
                                    fill=False, edgecolor='blue',
                                    linewidth=0.3, alpha=0.5)
                ax.add_patch(circle)
        if r_min_plot == 0:
            ax.add_patch(plt.Circle((0, 0), r_max_plot, fill=False,
                                     edgecolor='red', linewidth=1.5, linestyle='--'))
        else:
            for rv in [r_min_plot, r_max_plot]:
                ax.add_patch(plt.Circle((0, 0), rv, fill=False,
                                         edgecolor='red', linewidth=1.5, linestyle='--'))
        lim = r_max_plot * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')

    nn_dists, stats = nn_data[name.lower()]
    cv_str = f"CV={stats['cv']:.3f}" if stats else "n/a"
    ax.set_title(f'{name} (N={len(voxels)}, {cv_str})')
    ax.grid(True, alpha=0.3)

plt.suptitle('PMT-Positionen pro Area mit Homogenitäts-CV',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(path, "pmt_homogeneity_check.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"Plots gespeichert.")
print(f"Homogenitäts-Plot gespeichert.")


# =============================================================================
# Speichere selektierte Voxel
# =============================================================================

if geometry_name == "maxDist":
    selected_path = os.path.join(path, "homogeneous300PMTpositions_maxDist.json")
elif geometry_name == "currentDist":
    selected_path = os.path.join(path, "homogeneous300PMTpositions_currentDist.json")
else:
    selected_path = os.path.join(path, "homogeneous300PMTpositions.json")

with open(selected_path, "w") as f:
    json.dump(selected_voxels, f, indent=2)

print(f"\nSelektierte Voxel gespeichert in: {selected_path}")


# Plot the voxels using the new corner-based representation
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for voxel in all_voxels:
    # Extract the precomputed corners
    corners = voxel['corners']
    
    # Define the 6 faces using the corner points
    # Order: bottom, top, back, front, left, right
    faces = [
        [corners[0], corners[1], corners[5], corners[4]],  # bottom face
        [corners[3], corners[2], corners[6], corners[7]],  # top face
        [corners[0], corners[1], corners[2], corners[3]],  # back face (inner face)
        [corners[4], corners[5], corners[6], corners[7]],  # front face (outer face)
        [corners[0], corners[3], corners[7], corners[4]],  # left face
        [corners[1], corners[2], corners[6], corners[5]]   # right face
    ]
    
    # Add voxel to plot
    ax.add_collection3d(Poly3DCollection(
        faces, 
        facecolors='cyan', 
        edgecolors='k', 
        linewidths=0.3, 
        alpha=0.2
    ))

# Configure plot
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title(f'All Voxels for SSD')
ax.set_xlim(-r_zylinder*1.2, r_zylinder*1.2)
ax.set_ylim(-r_zylinder*1.2, r_zylinder*1.2)
ax.set_zlim(z_offset, z_offset + z_origin + h_zylinder + 10)

plt.tight_layout()
plt.show()


# Schaue ob zwei unterscheidliche Wall Arrays gleiche Index auf der gleichen Höhe haben
maxArray = voxels_wall

# In Dictionaries umwandeln für schnellen Zugriff
dict1 = {entry['index']: entry for entry in maxArray}
dict2 = {entry['index']: entry for entry in voxels_wall}

# Vergleiche gemeinsame Indices
common_indices = set(dict1.keys()) & set(dict2.keys())

warnungen = []

for idx in common_indices:
    center1 = dict1[idx].get('center', [])
    center2 = dict2[idx].get('center', [])
    
    if len(center1) < 3 or len(center2) < 3:
        warnungen.append(f"⚠️ Ungültiges 'center' bei index '{idx}'")
    elif center1[2] != center2[2]:
        warnungen.append(f"⚠️ Unterschied bei index '{idx}': center[2] unterschiedlich\n  array1: {center1[2]}\n  array2: {center2[2]}")

# Ausgabe der Unterschiede im center[2]
if warnungen:
    for warn in warnungen:
        print(warn)
else:
    print("✅ Alle gemeinsamen 'center[2]'-Werte sind gleich!")

# Prüfen, ob beide Arrays vollständig gleich sind
if maxArray == voxels_wall:
    print("⚠️ Die beiden Arrays sind vollständig identisch.")
else:
    print("✅ Die beiden Arrays sind unterschiedlich (mindestens ein Eintrag weicht ab).")
l Arrays gleiche Index auf der gleichen Höhe haben
maxArray = voxels_wall

# In Dictionaries umwandeln für schnellen Zugriff
dict1 = {entry['index']: entry for entry in maxArray}
dict2 = {entry['index']: entry for entry in voxels_wall}

# Vergleiche gemeinsame Indices
common_indices = set(dict1.keys()) & set(dict2.keys())

warnungen = []

for idx in common_indices:
    center1 = dict1[idx].get('center', [])
    center2 = dict2[idx].get('center', [])
    
    if len(center1) < 3 or len(center2) < 3:
        warnungen.append(f"⚠️ Ungültiges 'center' bei index '{idx}'")
    elif center1[2] != center2[2]:
        warnungen.append(f"⚠️ Unterschied bei index '{idx}': center[2] unterschiedlich\n  array1: {center1[2]}\n  array2: {center2[2]}")

# Ausgabe der Unterschiede im center[2]
if warnungen:
    for warn in warnungen:
        print(warn)
else:
    print("✅ Alle gemeinsamen 'center[2]'-Werte sind gleich!")

# Prüfen, ob beide Arrays vollständig gleich sind
if maxArray == voxels_wall:
    print("⚠️ Die beiden Arrays sind vollständig identisch.")
else:
    print("✅ Die beiden Arrays sind unterschiedlich (mindestens ein Eintrag weicht ab).")
