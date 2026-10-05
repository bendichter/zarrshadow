"""Load a zarrshadow JSON file and read data from the remote NWB file as zarr v3."""

from zarrshadow import open_rfs

# Open the reference file system as a zarr v3 group
root = open_rfs("examples/example.zarrshadow.json")

# Browse the hierarchy
print("Root attributes:")
for k, v in root.attrs.items():
    print(f"  {k}: {v}")

print()
print("Top-level groups:")
for name in root.group_keys():
    print(f"  {name}/")
for name in root.array_keys():
    print(f"  {name}")

# Read string scalar
print(f"\nidentifier: {root['identifier'][0]}")
print(f"session_description: {root['session_description'][0]}")

# Read numeric data (fetched from remote HDF5 via byte-range requests)
spike_times = root["units/spike_times"]
print(f"\nspike_times: shape={spike_times.shape}, dtype={spike_times.dtype}")
print(f"  first 10: {spike_times[:10]}")

# Read tabular data
start_time = root["intervals/trials/start_time"]
print(f"\ntrial start_time: shape={start_time.shape}")
print(f"  first 5: {start_time[:5]}")

# Read string array
locations = root["general/extracellular_ephys/electrodes/location"]
print(f"\nelectrode locations: {locations[:]}")

# Check soft link metadata
shank1 = root["general/extracellular_ephys/shank1"]
links = shank1.attrs.get("_LINKS", [])
print(f"\nshank1 _LINKS: {links}")
