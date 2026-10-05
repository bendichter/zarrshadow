"""Generate a zarrshadow reference file system (JSON) from a remote NWB file on DANDI."""

from zarrshadow import generate_rfs, write_rfs

url = "https://api.dandiarchive.org/api/assets/6e7e9b91-0d66-45af-b646-dfb11e4d9967/download/"

rfs = generate_rfs(url)

write_rfs(rfs, "examples/example.zarrshadow.json")
print(f"Wrote examples/example.zarrshadow.json ({len(rfs['refs'])} refs)")
