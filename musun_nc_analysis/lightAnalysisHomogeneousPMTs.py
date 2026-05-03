# Finde alle Files
print(f"\n1. Suche Files in {BASE_PATH}...")
all_files = find_all_files(BASE_PATH, RUNS)
print(f"   Gefunden: {len(all_files)} Files in {len(RUNS)} Runs")

# Zufällige Auswahl
if len(all_files) < NUM_FILES:
    print(f"   ⚠ Nur {len(all_files)} Files verfügbar, analysiere alle")
    selected_files = all_files
else:
    random.seed(42)  # Reproduzierbar
    selected_files = random.sample(all_files, NUM_FILES)

print(f"   → Analysiere {len(selected_files)} zufällig ausgewählte Files\n")

# =========================================================================
# TEIL 6: UMFANGREICHE Ge-77 ANALYSE
# =========================================================================

print(f"\n6. Erstelle umfangreiche Ge-77 Analyse...")

# Sammle Ge77-spezifische Daten
ge77_nc_x = []
ge77_nc_y = []
ge77_nc_z = []
ge77_nc_time = []
ge77_nc_tuples = []  # (evtid, nC_track_id)
ge77_photon_counts = []

# Für optische Photonen von Ge77-Events
ge77_optical_x = []
ge77_optical_y = []
ge77_optical_z = []
ge77_optical_time = []
ge77_optical_time_relative = []  # t - t_NC

print("   Sammle Ge-77 Event-Daten...")
for i, file_path in enumerate(selected_files, 1):
    if i % 20 == 0:
        print(f"   Progress: {i}/{len(selected_files)}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Lade NC-Daten
            nc_evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
            nc_track_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
            nc_x = f['hit']['MyNeutronCaptureOutput']['nC_x_position_in_m']['pages'][:]
            nc_y = f['hit']['MyNeutronCaptureOutput']['nC_y_position_in_m']['pages'][:]
            nc_z = f['hit']['MyNeutronCaptureOutput']['nC_z_position_in_m']['pages'][:]
            nc_time = f['hit']['MyNeutronCaptureOutput']['nC_time_in_ns']['pages'][:]
            nc_ge77_flags = f['hit']['MyNeutronCaptureOutput']['nC_flag_Ge77']['pages'][:]
            
            # Finde Ge77-Events
            ge77_mask = nc_ge77_flags > 0
            ge77_indices = np.where(ge77_mask)[0]
            
            if len(ge77_indices) == 0:
                continue
            
            # Speichere Ge77 NC-Positionen
            ge77_nc_x.extend(nc_x[ge77_mask])
            ge77_nc_y.extend(nc_y[ge77_mask])
            ge77_nc_z.extend(nc_z[ge77_mask])
            ge77_nc_time.extend(nc_time[ge77_mask])
            
            # Erstelle Dictionary: (evtid, nC_track_id) -> NC-Zeit
            ge77_nc_dict = {}
            for idx in ge77_indices:
                nc_tuple = (nc_evtid[idx], nc_track_id[idx])
                ge77_nc_tuples.append(nc_tuple)
                ge77_nc_dict[nc_tuple] = nc_time[idx]
            
           # Lade optische Photonen
            optical_evtid = f['hit']['optical']['evtid']['pages'][:]
            optical_nc_track_id = f['hit']['optical']['nC_track_id']['pages'][:]
            optical_det_uid = f['hit']['optical']['det_uid']['pages'][:]  
            optical_x = f['hit']['optical']['x_position_in_m']['pages'][:]
            optical_y = f['hit']['optical']['y_position_in_m']['pages'][:]
            optical_z = f['hit']['optical']['z_position_in_m']['pages'][:]
            optical_time = f['hit']['optical']['time_in_ns']['pages'][:]

            # Filter: Nur Tyvek-Detektoren
            valid_detector_mask = np.isin(optical_det_uid, [1965, 1966, 1967, 1968])  

            # Filter: Nur Photonen von Ge77-Events UND Tyvek-Detektoren
            optical_tuples = list(zip(optical_evtid[valid_detector_mask], 
                                    optical_nc_track_id[valid_detector_mask]))  

            optical_x_filtered = optical_x[valid_detector_mask]  
            optical_y_filtered = optical_y[valid_detector_mask]  
            optical_z_filtered = optical_z[valid_detector_mask]  
            optical_time_filtered = optical_time[valid_detector_mask]  

            for j, opt_tuple in enumerate(optical_tuples):
                if opt_tuple in ge77_nc_dict:
                    ge77_optical_x.append(optical_x_filtered[j])  
                    ge77_optical_y.append(optical_y_filtered[j])  
                    ge77_optical_z.append(optical_z_filtered[j])  
                    ge77_optical_time.append(optical_time_filtered[j])  
                    
                    # Relative Zeit: t_photon - t_NC
                    nc_time_ref = ge77_nc_dict[opt_tuple]
                    ge77_optical_time_relative.append(optical_time[j] - nc_time_ref)
            
            # Zähle Photonen pro Ge77-NC
            ge77_photon_counter = Counter(optical_tuples)
            for nc_tuple in ge77_nc_dict.keys():
                photon_count = ge77_photon_counter.get(nc_tuple, 0)
                ge77_photon_counts.append(photon_count)
    
    except Exception as e:
        print(f"   ⚠ Fehler bei {file_path.name}: {e}")
        continue

# Arrays konvertieren
ge77_nc_x = np.array(ge77_nc_x)
ge77_nc_y = np.array(ge77_nc_y)
ge77_nc_z = np.array(ge77_nc_z)
ge77_nc_time = np.array(ge77_nc_time)
ge77_photon_counts = np.array(ge77_photon_counts)

ge77_optical_x = np.array(ge77_optical_x)
ge77_optical_y = np.array(ge77_optical_y)
ge77_optical_z = np.array(ge77_optical_z)
ge77_optical_time = np.array(ge77_optical_time)
ge77_optical_time_relative = np.array(ge77_optical_time_relative)

total_ge77_nc = len(ge77_nc_x)
ge77_with_photons = np.sum(ge77_photon_counts > 0)
ge77_without_photons = total_ge77_nc - ge77_with_photons
total_ge77_photons = len(ge77_optical_x)

print(f"   ✓ Ge-77 Analyse:")
print(f"     Total Ge-77 NC-Events: {total_ge77_nc:,}")
print(f"     Mit optischen Photonen: {ge77_with_photons:,} ({ge77_with_photons/total_ge77_nc*100:.2f}%)")
print(f"     Ohne optische Photonen: {ge77_without_photons:,}")
print(f"     Total optische Photonen von Ge-77: {total_ge77_photons:,}\n")

if len(ge77_optical_x) > 0:
    # DEBUG
    print(f"   DEBUG: Plotte {len(ge77_optical_x):,} Photonen")
    print(f"   DEBUG: X range: {ge77_optical_x.min():.3f} bis {ge77_optical_x.max():.3f}")
    print(f"   DEBUG: Y range: {ge77_optical_y.min():.3f} bis {ge77_optical_y.max():.3f}")
    print(f"   DEBUG: Z range: {ge77_optical_z.min():.3f} bis {ge77_optical_z.max():.3f}")
    
    # Sample für Performance (falls >100k Photonen)

# =========================================================================
# PLOT 1: 3D Positionen der Ge77 NC-Events
# =========================================================================

print("   Erstelle Plot 1: 3D Positionen Ge-77 NC-Events...")

fig1 = plt.figure(figsize=(14, 10))
ax1 = fig1.add_subplot(111, projection='3d')

# Farbcode: Mit/Ohne Photonen
colors = ['gold' if c > 0 else 'gray' for c in ge77_photon_counts]

scatter = ax1.scatter(ge77_nc_x, ge77_nc_y, ge77_nc_z, 
                        c=colors, s=20, alpha=0.6, edgecolors='black', linewidths=0.5)

ax1.set_xlabel('X Position [m]', fontsize=12)
ax1.set_ylabel('Y Position [m]', fontsize=12)
ax1.set_zlabel('Z Position [m]', fontsize=12)
ax1.set_title(f'3D Positionen: Ge-77 Neutron Captures\n({total_ge77_nc:,} Events)', 
                fontsize=16, fontweight='bold')

# Legende
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
            markersize=10, label=f'Mit Photonen ({ge77_with_photons:,})'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
            markersize=10, label=f'Ohne Photonen ({ge77_without_photons:,})')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Zylinder-Umriss hinzufügen
theta = np.linspace(0, 2*np.pi, 100)
z_cyl = np.array([-5.0, 4.3])  # -5000mm bis 4300mm in m
r_cyl = 4.5  # 4500mm in m

for z_val in z_cyl:
    x_cyl = r_cyl * np.cos(theta)
    y_cyl = r_cyl * np.sin(theta)
    z_cyl_line = np.full_like(x_cyl, z_val)
    ax1.plot(x_cyl, y_cyl, z_cyl_line, 'r--', linewidth=2, alpha=0.7)

# Vertikale Linien
for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
    x_vert = r_cyl * np.cos(angle)
    y_vert = r_cyl * np.sin(angle)
    ax1.plot([x_vert, x_vert], [y_vert, y_vert], [-5.0, 4.3], 'r--', linewidth=1, alpha=0.5)

plt.tight_layout()
output_ge77_1 = Path.cwd() / "ge77_nc_positions_3d.png"
plt.savefig(output_ge77_1, dpi=300, bbox_inches='tight')
print(f"   ✓ Gespeichert: {output_ge77_1}")

# =========================================================================
# PLOT 2: Ge77-Events mit/ohne Licht (Pie Chart)
# =========================================================================

print("   Erstelle Plot 2: Ge77 mit/ohne optische Photonen...")

fig2, ax2 = plt.subplots(figsize=(10, 8))

wedges, texts, autotexts = ax2.pie([ge77_with_photons, ge77_without_photons],
                                    labels=['Mit Photonen', 'Ohne Photonen'],
                                    autopct='%1.2f%%',
                                    colors=['gold', 'lightgray'],
                                    explode=(0.05, 0),
                                    startangle=90,
                                    textprops={'fontsize': 13, 'fontweight': 'bold'},
                                    wedgeprops={'linewidth': 2, 'edgecolor': 'black'})

ax2.set_title(f'Ge-77 NC-Events: Lichtproduktion\n({total_ge77_nc:,} Events)',
                fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
output_ge77_2 = Path.cwd() / "ge77_light_production.png"
plt.savefig(output_ge77_2, dpi=300, bbox_inches='tight')
print(f"   ✓ Gespeichert: {output_ge77_2}")

# =========================================================================
# PLOT 3: Histogramm Photonen-Anzahl (nur Events mit Photonen)
# =========================================================================

print("   Erstelle Plot 3: Histogramm Photonen-Anzahl...")

fig3, ax3 = plt.subplots(figsize=(12, 8))

ge77_photon_counts_nonzero = ge77_photon_counts[ge77_photon_counts > 0]

if len(ge77_photon_counts_nonzero) > 0:
    bins = np.logspace(0, np.log10(ge77_photon_counts_nonzero.max()), 40)
    
    ax3.hist(ge77_photon_counts_nonzero, bins=bins, alpha=0.7, color='gold',
            edgecolor='black', linewidth=1.5)
    
    ax3.set_xlabel('Anzahl optischer Photonen', fontsize=14)
    ax3.set_ylabel('Anzahl Ge-77 NC-Events', fontsize=14)
    ax3.set_title(f'Ge-77: Verteilung optischer Photonen\n(nur Events mit >0 Photonen, n={len(ge77_photon_counts_nonzero):,})',
                    fontsize=16, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Statistiken
    mean_photons = np.mean(ge77_photon_counts_nonzero)
    median_photons = np.median(ge77_photon_counts_nonzero)
    
    ax3.axvline(mean_photons, color='red', linestyle='--', linewidth=2,
                label=f'Mittelwert: {mean_photons:.1f}')
    ax3.axvline(median_photons, color='darkred', linestyle=':', linewidth=2,
                label=f'Median: {median_photons:.0f}')
    ax3.legend(fontsize=12)

plt.tight_layout()
output_ge77_3 = Path.cwd() / "ge77_photon_count_histogram.png"
plt.savefig(output_ge77_3, dpi=300, bbox_inches='tight')
print(f"   ✓ Gespeichert: {output_ge77_3}")

# =========================================================================
# PLOT 4: Zeitverteilung optischer Photonen (relativ zu NC-Zeit)
# =========================================================================

print("   Erstelle Plot 4: Zeitverteilung optischer Photonen...")

fig4, ax4 = plt.subplots(figsize=(12, 8))

if len(ge77_optical_time_relative) > 0:
    # Bins: -50 ns bis +500 ns (relative Zeit)
    time_bins = np.linspace(-50, 500, 100)
    
    ax4.hist(ge77_optical_time_relative, bins=time_bins, alpha=0.7, color='cyan',
            edgecolor='black', linewidth=1.5)
    
    ax4.set_xlabel('Zeit relativ zu NC [ns]', fontsize=14)
    ax4.set_ylabel('Anzahl optischer Photonen', fontsize=14)
    ax4.set_title(f'Ge-77: Zeitverteilung optischer Photonen\n(t=0 bei Neutron Capture, n={len(ge77_optical_time_relative):,} Photonen)',
                    fontsize=16, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='NC-Zeitpunkt')
    
    # Statistiken
    mean_time = np.mean(ge77_optical_time_relative)
    median_time = np.median(ge77_optical_time_relative)
    
    ax4.axvline(mean_time, color='darkblue', linestyle='--', linewidth=2,
                label=f'Mittelwert: {mean_time:.1f} ns')
    ax4.axvline(median_time, color='purple', linestyle=':', linewidth=2,
                label=f'Median: {median_time:.1f} ns')
    ax4.legend(fontsize=12)

    # Berechne Prozent vor 200ns
    photons_before_200ns = np.sum(ge77_optical_time_relative < 200)
    percent_before_200ns = (photons_before_200ns / len(ge77_optical_time_relative) * 100)
    
    # Text hinzufügen
    ax4.text(0.98, 0.02, f'{percent_before_200ns:.1f}% der Photonen vor 200ns',
            transform=ax4.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
output_ge77_4 = Path.cwd() / "ge77_photon_time_distribution.png"
plt.savefig(output_ge77_4, dpi=300, bbox_inches='tight')
print(f"   ✓ Gespeichert: {output_ge77_4}")

# =========================================================================
# PLOT 5: 3D Positionen optischer Photonen von Ge77-Events
# =========================================================================

print("   Erstelle Plot 5: 3D Positionen optischer Photonen...")

fig5 = plt.figure(figsize=(14, 10))
ax5 = fig5.add_subplot(111, projection='3d')

if len(ge77_optical_x) > 0:
    # Sample für Performance (falls >100k Photonen)
    if len(ge77_optical_x) > 100000:
        sample_indices = np.random.choice(len(ge77_optical_x), 100000, replace=False)
        plot_x = ge77_optical_x[sample_indices]
        plot_y = ge77_optical_y[sample_indices]
        plot_z = ge77_optical_z[sample_indices]
        sample_note = f"\n(Sample: 100k von {len(ge77_optical_x):,})"
    else:
        plot_x = ge77_optical_x
        plot_y = ge77_optical_y
        plot_z = ge77_optical_z
        sample_note = ""
    
    scatter = ax5.scatter(plot_x, plot_y, plot_z,
                        c='red', s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    print(f"   DEBUG: Scatter erstellt mit {len(plot_x)} Punkten")
    
    ax5.set_xlabel('X Position [m]', fontsize=12)
    ax5.set_ylabel('Y Position [m]', fontsize=12)
    ax5.set_zlabel('Z Position [m]', fontsize=12)
    ax5.set_title(f'3D Positionen: Optische Photonen von Ge-77 Events{sample_note}',
                    fontsize=16, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Dynamische Limits basierend auf Daten
    x_margin = (ge77_optical_x.max() - ge77_optical_x.min()) * 0.1
    y_margin = (ge77_optical_y.max() - ge77_optical_y.min()) * 0.1
    z_margin = (ge77_optical_z.max() - ge77_optical_z.min()) * 0.1
    
    ax5.set_xlim([ge77_optical_x.min() - x_margin, ge77_optical_x.max() + x_margin])
    ax5.set_ylim([ge77_optical_y.min() - y_margin, ge77_optical_y.max() + y_margin])
    ax5.set_zlim([ge77_optical_z.min() - z_margin, ge77_optical_z.max() + z_margin])

    # Zylinder-Umriss hinzufügen
    theta = np.linspace(0, 2*np.pi, 100)
    z_cyl = np.array([-5.0, 4.3])
    r_cyl = 4.5
    
    for z_val in z_cyl:
        x_cyl = r_cyl * np.cos(theta)
        y_cyl = r_cyl * np.sin(theta)
        z_cyl_line = np.full_like(x_cyl, z_val)
        ax5.plot(x_cyl, y_cyl, z_cyl_line, 'r--', linewidth=2, alpha=0.7)
    
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_vert = r_cyl * np.cos(angle)
        y_vert = r_cyl * np.sin(angle)
        ax5.plot([x_vert, x_vert], [y_vert, y_vert], [-5.0, 4.3], 'r--', linewidth=1, alpha=0.5)

plt.tight_layout()
output_ge77_5 = Path.cwd() / "ge77_photon_positions_3d.png"
plt.savefig(output_ge77_5, dpi=300, bbox_inches='tight')
print(f"   ✓ Gespeichert: {output_ge77_5}")

# =========================================================================
# Ge77-Statistiken speichern
# =========================================================================

print("\n   Schreibe Ge-77 Statistiken...")

stats_ge77_file = Path.cwd() / "ge77_statistics.txt"
with open(stats_ge77_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("Ge-77 NEUTRON CAPTURE STATISTIKEN\n")
    f.write("="*60 + "\n\n")
    
    f.write("NEUTRON CAPTURES:\n")
    f.write(f"  Total Ge-77 NC-Events: {total_ge77_nc:,}\n")
    f.write(f"  Mit optischen Photonen: {ge77_with_photons:,} ({ge77_with_photons/total_ge77_nc*100:.2f}%)\n")
    f.write(f"  Ohne optische Photonen: {ge77_without_photons:,} ({ge77_without_photons/total_ge77_nc*100:.2f}%)\n\n")
    
    f.write("POSITIONEN (NC-Events):\n")
    f.write(f"  X: min={ge77_nc_x.min():.3f} m, max={ge77_nc_x.max():.3f} m\n")
    f.write(f"  Y: min={ge77_nc_y.min():.3f} m, max={ge77_nc_y.max():.3f} m\n")
    f.write(f"  Z: min={ge77_nc_z.min():.3f} m, max={ge77_nc_z.max():.3f} m\n\n")
    
    f.write("OPTISCHE PHOTONEN:\n")
    f.write(f"  Total Photonen: {total_ge77_photons:,}\n")
    
    if len(ge77_photon_counts_nonzero) > 0:
        f.write(f"  Photonen/NC (nur >0):\n")
        f.write(f"    Mittelwert: {np.mean(ge77_photon_counts_nonzero):.2f}\n")
        f.write(f"    Median: {np.median(ge77_photon_counts_nonzero):.0f}\n")
        f.write(f"    Std Dev: {np.std(ge77_photon_counts_nonzero):.2f}\n")
        f.write(f"    Min: {np.min(ge77_photon_counts_nonzero):,}\n")
        f.write(f"    Max: {np.max(ge77_photon_counts_nonzero):,}\n\n")
    
    if len(ge77_optical_time_relative) > 0:
        f.write("ZEITVERTEILUNG (relativ zu NC):\n")
        f.write(f"  Mittelwert: {np.mean(ge77_optical_time_relative):.2f} ns\n")
        f.write(f"  Median: {np.median(ge77_optical_time_relative):.2f} ns\n")
        f.write(f"  Std Dev: {np.std(ge77_optical_time_relative):.2f} ns\n")
        f.write(f"  Min: {np.min(ge77_optical_time_relative):.2f} ns\n")
        f.write(f"  Max: {np.max(ge77_optical_time_relative):.2f} ns\n\n")
    
    f.write("PHOTONEN-POSITIONEN:\n")
    f.write(f"  X: min={ge77_optical_x.min():.3f} m, max={ge77_optical_x.max():.3f} m\n")
    f.write(f"  Y: min={ge77_optical_y.min():.3f} m, max={ge77_optical_y.max():.3f} m\n")
    f.write(f"  Z: min={ge77_optical_z.min():.3f} m, max={ge77_optical_z.max():.3f} m\n")

print(f"   ✓ Ge-77 Statistiken gespeichert: {stats_ge77_file}\n")

print(f"{'='*60}")
print(f"Ge-77 ANALYSE ABGESCHLOSSEN!")
print(f"{'='*60}")
print(f"\nGespeicherte Plots:")
print(f"  1. {output_ge77_1.name}")
print(f"  2. {output_ge77_2.name}")
print(f"  3. {output_ge77_3.name}")
print(f"  4. {output_ge77_4.name}")
print(f"  5. {output_ge77_5.name}")
print(f"\nStatistik-Datei: {stats_ge77_file.name}\n")

plt.show()

# =========================================================================
# PLOT 5: Heatmaps optischer Photonen auf Detektorflächen
# =========================================================================

print("   Erstelle Plot 5: Heatmaps auf Detektorflächen...")

# Lade detector UIDs für Ge77-Photonen
ge77_optical_det_uid = []

print("   Lade Detektor-IDs für Ge77-Photonen...")
for i, file_path in enumerate(selected_files, 1):
    try:
        with h5py.File(file_path, 'r') as f:
            # NC-Daten für Ge77-Filter
            nc_evtid = f['hit']['MyNeutronCaptureOutput']['evtid']['pages'][:]
            nc_track_id = f['hit']['MyNeutronCaptureOutput']['nC_track_id']['pages'][:]
            nc_ge77_flags = f['hit']['MyNeutronCaptureOutput']['nC_flag_Ge77']['pages'][:]
            
            ge77_mask = nc_ge77_flags > 0
            ge77_indices = np.where(ge77_mask)[0]
            
            if len(ge77_indices) == 0:
                continue
            
            ge77_nc_set = set(zip(nc_evtid[ge77_mask], nc_track_id[ge77_mask]))
            
            # Optische Photonen
            optical_evtid = f['hit']['optical']['evtid']['pages'][:]
            optical_nc_track_id = f['hit']['optical']['nC_track_id']['pages'][:]
            optical_det_uid = f['hit']['optical']['det_uid']['pages'][:]
            
            optical_tuples = list(zip(optical_evtid, optical_nc_track_id))
            
            for j, opt_tuple in enumerate(optical_tuples):
                if opt_tuple in ge77_nc_set:
                    ge77_optical_det_uid.append(optical_det_uid[j])
    
    except Exception as e:
        continue

ge77_optical_det_uid = np.array(ge77_optical_det_uid)

# Filter Photonen nach Detektor-ID
tyvek_zyl_mask = ge77_optical_det_uid == 1965
tyvek_bot_mask = ge77_optical_det_uid == 1967
tyvek_top_mask = ge77_optical_det_uid == 1968
tyvek_pit_mask = ge77_optical_det_uid == 1966

# Positionen für jede Fläche
zyl_x = ge77_optical_x[tyvek_zyl_mask]
zyl_y = ge77_optical_y[tyvek_zyl_mask]
zyl_z = ge77_optical_z[tyvek_zyl_mask]

bot_x = ge77_optical_x[tyvek_bot_mask]
bot_y = ge77_optical_y[tyvek_bot_mask]

top_x = ge77_optical_x[tyvek_top_mask]
top_y = ge77_optical_y[tyvek_top_mask]

pit_x = ge77_optical_x[tyvek_pit_mask]
pit_y = ge77_optical_y[tyvek_pit_mask]

print(f"   Photonen-Verteilung:")
print(f"     Zylinder (1965): {len(zyl_x):,}")
print(f"     Boden (1967): {len(bot_x):,}")
print(f"     Deckel (1968): {len(top_x):,}")
print(f"     Pit (1966): {len(pit_x):,}")

# Erstelle 2x2 Grid
fig5, axes = plt.subplots(2, 2, figsize=(16, 14))

# ===== Plot 5a: Zylinder (phi vs z) =====
ax_zyl = axes[0, 0]

if len(zyl_x) > 0:
    # Berechne phi in Zylinderkoordinaten
    zyl_phi = np.arctan2(zyl_y, zyl_x)  # -pi bis pi
    zyl_phi_deg = np.degrees(zyl_phi)  # -180 bis 180
    
    # Bins
    phi_bins = np.linspace(-180, 180, 100)
    z_bins = np.linspace(zyl_z.min(), zyl_z.max(), 100)
    
    hist_zyl, phi_edges, z_edges = np.histogram2d(zyl_phi_deg, zyl_z, 
                                                    bins=[phi_bins, z_bins])
    
    im_zyl = ax_zyl.imshow(hist_zyl.T, origin='lower', aspect='auto',
                            extent=[phi_edges[0], phi_edges[-1], z_edges[0], z_edges[-1]],
                            cmap='hot', norm=plt.matplotlib.colors.LogNorm(vmin=1))
    
    ax_zyl.set_xlabel('φ [°]', fontsize=12)
    ax_zyl.set_ylabel('z [m]', fontsize=12)
    ax_zyl.set_title(f'Tyvek Zylinder (det_uid=1965)\n{len(zyl_x):,} Photonen', 
                    fontsize=13, fontweight='bold')
    plt.colorbar(im_zyl, ax=ax_zyl, label='Hits (log)')
    ax_zyl.grid(True, alpha=0.3, color='white')
else:
    ax_zyl.text(0.5, 0.5, 'Keine Photonen', ha='center', va='center', fontsize=14)
    ax_zyl.set_title('Tyvek Zylinder (det_uid=1965)', fontsize=13, fontweight='bold')

# ===== Plot 5b: Boden (x vs y) =====
ax_bot = axes[0, 1]

if len(bot_x) > 0:
    x_bins = np.linspace(bot_x.min(), bot_x.max(), 100)
    y_bins = np.linspace(bot_y.min(), bot_y.max(), 100)
    
    hist_bot, x_edges, y_edges = np.histogram2d(bot_x, bot_y, 
                                                    bins=[x_bins, y_bins])
    
    im_bot = ax_bot.imshow(hist_bot.T, origin='lower', aspect='auto',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            cmap='hot', norm=plt.matplotlib.colors.LogNorm(vmin=1))
    
    ax_bot.set_xlabel('x [m]', fontsize=12)
    ax_bot.set_ylabel('y [m]', fontsize=12)
    ax_bot.set_title(f'Tyvek Boden (det_uid=1967)\n{len(bot_x):,} Photonen', 
                    fontsize=13, fontweight='bold')
    plt.colorbar(im_bot, ax=ax_bot, label='Hits (log)')
    ax_bot.grid(True, alpha=0.3, color='white')
else:
    ax_bot.text(0.5, 0.5, 'Keine Photonen', ha='center', va='center', fontsize=14)
    ax_bot.set_title('Tyvek Boden (det_uid=1967)', fontsize=13, fontweight='bold')

# ===== Plot 5c: Deckel (x vs y) =====
ax_top = axes[1, 0]

if len(top_x) > 0:
    x_bins = np.linspace(top_x.min(), top_x.max(), 100)
    y_bins = np.linspace(top_y.min(), top_y.max(), 100)
    
    hist_top, x_edges, y_edges = np.histogram2d(top_x, top_y, 
                                                    bins=[x_bins, y_bins])
    
    im_top = ax_top.imshow(hist_top.T, origin='lower', aspect='auto',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            cmap='hot', norm=plt.matplotlib.colors.LogNorm(vmin=1))
    
    ax_top.set_xlabel('x [m]', fontsize=12)
    ax_top.set_ylabel('y [m]', fontsize=12)
    ax_top.set_title(f'Tyvek Deckel (det_uid=1968)\n{len(top_x):,} Photonen', 
                    fontsize=13, fontweight='bold')
    plt.colorbar(im_top, ax=ax_top, label='Hits (log)')
    ax_top.grid(True, alpha=0.3, color='white')
else:
    ax_top.text(0.5, 0.5, 'Keine Photonen', ha='center', va='center', fontsize=14)
    ax_top.set_title('Tyvek Deckel (det_uid=1968)', fontsize=13, fontweight='bold')

# ===== Plot 5d: Pit (x vs y) =====
ax_pit = axes[1, 1]

if len(pit_x) > 0:
    x_bins = np.linspace(pit_x.min(), pit_x.max(), 100)
    y_bins = np.linspace(pit_y.min(), pit_y.max(), 100)
    
    hist_pit, x_edges, y_edges = np.histogram2d(pit_x, pit_y, 
                                                    bins=[x_bins, y_bins])
    
    im_pit = ax_pit.imshow(hist_pit.T, origin='lower', aspect='auto',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            cmap='hot', norm=plt.matplotlib.colors.LogNorm(vmin=1))
    
    ax_pit.set_xlabel('x [m]', fontsize=12)
    ax_pit.set_ylabel('y [m]', fontsize=12)
    ax_pit.set_title(f'Tyvek Pit (det_uid=1966)\n{len(pit_x):,} Photonen', 
                    fontsize=13, fontweight='bold')
    plt.colorbar(im_pit, ax=ax_pit, label='Hits (log)')
    ax_pit.grid(True, alpha=0.3, color='white')
else:
    ax_pit.text(0.5, 0.5, 'Keine Photonen', ha='center', va='center', fontsize=14)
    ax_pit.set_title('Tyvek Pit (det_uid=1966)', fontsize=13, fontweight='bold')

plt.tight_layout()
output_ge77_5 = Path.cwd() / "ge77_detector_heatmaps.png"
plt.savefig(output_ge77_5, dpi=300, bbox_inches='tight')
print(f"   ✓ Gespeichert: {output_ge77_5}")