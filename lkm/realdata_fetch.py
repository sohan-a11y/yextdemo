"""Fetch a spatial slice of Overture Maps Places from the public S3 bucket.

Uses pyarrow's native S3 filesystem (no DuckDB extension needed) and prunes parquet
row groups with their bbox statistics, so only the rows near the requested region
are downloaded.
"""
import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.fs as pafs

BUCKET = "overturemaps-us-west-2"
COLUMNS = ["id", "bbox", "confidence", "names", "sources", "operating_status",
           "basic_category", "websites", "phones", "addresses", "brand"]
NYC_BBOX = (-74.05, 40.54, -73.70, 40.92)   # xmin, ymin, xmax, ymax


def make_fs():
    kw = dict(anonymous=True, region="us-west-2")
    proxy = os.environ.get("HTTPS_PROXY", "")
    if proxy:
        host, port = proxy.replace("http://", "").rstrip("/").split(":")
        kw["proxy_options"] = {"scheme": "http", "host": host, "port": int(port)}
    ca = "/root/.ccr/ca-bundle.crt"
    if os.path.exists(ca):
        kw["tls_ca_file_path"] = ca
    return pafs.S3FileSystem(**kw)


def list_files(fs, release):
    path = f"{BUCKET}/release/{release}/theme=places/type=place/"
    infos = fs.get_file_info(pafs.FileSelector(path, recursive=False))
    return sorted(i.path for i in infos if i.size and i.size > 0)


def _row_groups_for(fs, f, bbox):
    pf = pq.ParquetFile(f, filesystem=fs)
    md = pf.metadata
    names = [md.row_group(0).column(j).path_in_schema for j in range(md.num_columns)]
    ix = {n: names.index(n) for n in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}
    keep = []
    X0, Y0, X1, Y1 = bbox
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        s = {k: rg.column(v).statistics for k, v in ix.items()}
        if any(v is None or not v.has_min_max for v in s.values()):
            keep.append(g); continue
        if s["bbox.xmin"].min <= X1 and s["bbox.xmax"].max >= X0 and \
           s["bbox.ymin"].min <= Y1 and s["bbox.ymax"].max >= Y0:
            keep.append(g)
    return pf, keep


def _read_file(args):
    fs, f, bbox = args
    pf, keep = _row_groups_for(fs, f, bbox)
    if not keep:
        return None, 0, 0
    tb = pf.read_row_groups(keep, columns=COLUMNS)
    return tb, len(keep), pf.metadata.num_row_groups


def fetch_release(release, bbox=NYC_BBOX, out_dir="lkm_data", workers=8, log=print):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"overture_{release}_nyc.parquet")
    if os.path.exists(out):
        log(f"[fetch] cache hit {out}")
        return out
    fs = make_fs()
    files = list_files(fs, release)
    log(f"[fetch] {release}: {len(files)} files; pruning row groups to bbox {bbox}")
    t0 = time.time(); tables = []; kept = total = 0
    with ThreadPoolExecutor(workers) as ex:
        for tb, k, n in ex.map(_read_file, [(fs, f, bbox) for f in files]):
            kept += k; total += n
            if tb is not None and tb.num_rows:
                tables.append(tb)
    log(f"[fetch] read {kept}/{total} row groups in {time.time()-t0:.0f}s")
    tb = pa.concat_tables(tables, promote_options="default")
    bx = tb.column("bbox").combine_chunks()
    xmin = bx.field("xmin").to_numpy(zero_copy_only=False); ymin = bx.field("ymin").to_numpy(zero_copy_only=False)
    X0, Y0, X1, Y1 = bbox
    mask = (xmin >= X0) & (xmin <= X1) & (ymin >= Y0) & (ymin <= Y1)
    tb = tb.filter(pa.array(mask))
    pq.write_table(tb, out, compression="zstd")
    log(f"[fetch] {release}: {tb.num_rows} places in region -> {out} ({os.path.getsize(out)/1e6:.0f} MB)")
    return out


if __name__ == "__main__":
    for r in sys.argv[1:] or ["2026-08-19.0", "2026-09-23.0"]:
        fetch_release(r)
