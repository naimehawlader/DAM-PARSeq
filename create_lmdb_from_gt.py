import os
import lmdb
import cv2


def create_lmdb_from_gt(image_dir, gt_file, output_path):

    os.makedirs(output_path, exist_ok=True)

    # safer LMDB config
    env = lmdb.open(
        output_path,
        map_size=1099511627776,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = env.begin(write=True)

    samples = 0

    with open(gt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue

        img_name, label = parts
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            print("Missing:", img_path)
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        _, img_encoded = cv2.imencode(".jpg", img)

        # normalize label
        label = label.strip()

        # IMPORTANT: 1-based indexing (PARSeq expects this style)
        idx = samples + 1

        image_key = f"image-{idx:09d}".encode()
        label_key = f"label-{idx:09d}".encode()

        txn.put(image_key, img_encoded.tobytes())
        txn.put(label_key, label.encode())

        samples += 1

    # commit properly
    txn.put(b"num-samples", str(samples).encode())
    txn.commit()
    env.close()

    print("LMDB created at:", output_path)
    print("Total samples:", samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    create_lmdb_from_gt(args.img_dir, args.gt, args.output)