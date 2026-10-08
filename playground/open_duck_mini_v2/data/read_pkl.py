import pickle
import numpy as np
import pandas as pd
import itertools
from pathlib import Path


def export_prm_pkl_to_excel(pkl_path: str, excel_path: str) -> None:
    """
    PRM 스타일 pkl 파일을 읽어서, 내용을 엑셀로 저장한다.
    예상 구조:
        {
            "dxs": 1D array-like,
            "dys": 1D array-like,
            "dthetas": 1D array-like,
            "nb_steps_in_period": int,
            "data_array": ndarray with shape
                (len(dxs), len(dys), len(dthetas), num_features, poly_degree+1)
        }
    """
    pkl_path = Path(pkl_path)
    excel_path = Path(excel_path)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 엑셀 writer 준비
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # 1) 메타 정보 요약
        meta_rows = []

        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    meta_rows.append(
                        {
                            "key": k,
                            "type": "ndarray",
                            "dtype": str(v.dtype),
                            "shape": str(v.shape),
                        }
                    )
                elif isinstance(v, (list, tuple)):
                    meta_rows.append(
                        {
                            "key": k,
                            "type": type(v).__name__,
                            "dtype": "mixed",
                            "shape": f"(len={len(v)})",
                        }
                    )
                else:
                    meta_rows.append(
                        {
                            "key": k,
                            "type": type(v).__name__,
                            "dtype": "",
                            "shape": "",
                            "value_preview": str(v)[:80],
                        }
                    )
        else:
            meta_rows.append(
                {
                    "key": "root",
                    "type": type(data).__name__,
                    "dtype": "",
                    "shape": "",
                    "value_preview": str(data)[:80],
                }
            )

        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_excel(writer, sheet_name="meta", index=False)

        # 2) PRM 스타일이면 grids와 coeffs를 별도로 풀어주기
        if isinstance(data, dict) and all(k in data for k in ["dxs", "dys", "dthetas", "data_array"]):
            dxs = np.array(data["dxs"])
            dys = np.array(data["dys"])
            dthetas = np.array(data["dthetas"])
            data_array = np.array(data["data_array"])
            nb_steps_in_period = data.get("nb_steps_in_period", None)

            # vel_grids 시트: dxs, dys, dthetas
            max_len = max(len(dxs), len(dys), len(dthetas))
            def pad(arr, target_len):
                out = np.full(target_len, np.nan)
                out[: len(arr)] = arr
                return out

            vel_df = pd.DataFrame({
                "dxs": pad(dxs, max_len),
                "dys": pad(dys, max_len),
                "dthetas": pad(dthetas, max_len),
            })

            if nb_steps_in_period is not None:
                vel_df.attrs["nb_steps_in_period"] = nb_steps_in_period

            vel_df.to_excel(writer, sheet_name="vel_grids", index=False)

            # 3) coeffs_flat: (ix, iy, itheta, feature, power, coeff)
            # data_array.shape = (nx, ny, nt, n_feat, deg+1)
            if data_array.ndim != 5:
                print(f"[WARNING] data_array ndim={data_array.ndim}, expected 5. Skipping coeffs_flat.")
            else:
                nx, ny, nt, n_feat, n_coeff = data_array.shape

                rows = []
                # 너무 크면 부담될 수 있으니, 필요하면 여기서 일부만 샘플링하도록 수정 가능
                for ix, iy, itheta in itertools.product(range(nx), range(ny), range(nt)):
                    coeff_block = data_array[ix, iy, itheta]  # shape (n_feat, n_coeff)
                    for feat_idx in range(n_feat):
                        for p_idx in range(n_coeff):
                            rows.append(
                                {
                                    "ix": ix,
                                    "iy": iy,
                                    "itheta": itheta,
                                    "feature": feat_idx,
                                    # power: jp.polyval에서 사용하는 순서에 맞게, 보통 highest degree -> ... -> constant
                                    "power_index": p_idx,
                                    "coeff": float(coeff_block[feat_idx, p_idx]),
                                }
                            )

                coeffs_df = pd.DataFrame(rows)
                coeffs_df.to_excel(writer, sheet_name="coeffs_flat", index=False)

        else:
            # PRM 구조가 아니면, 가능한 것만 단순 덤프
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        if isinstance(v, np.ndarray):
                            df = pd.DataFrame(v.reshape(-1, v.shape[-1])) if v.ndim >= 2 else pd.DataFrame(v)
                        elif isinstance(v, (list, tuple)):
                            df = pd.DataFrame(v)
                        else:
                            df = pd.DataFrame({"value": [v]})
                        safe_name = str(k)[:30]
                        df.to_excel(writer, sheet_name=safe_name, index=False)
                    except Exception as e:
                        print(f"[WARNING] Could not write key '{k}' to its own sheet: {e}")
            else:
                # data 자체가 ndarray나 list인 경우
                if isinstance(data, np.ndarray):
                    df = pd.DataFrame(data.reshape(-1, data.shape[-1])) if data.ndim >= 2 else pd.DataFrame(data)
                elif isinstance(data, (list, tuple)):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame({"value": [data]})

                df.to_excel(writer, sheet_name="data", index=False)

    print(f"Exported to {excel_path}")


if __name__ == "__main__":
    # 사용 예시
    # python script.py 처럼 쓸 거면 argparse 추가하면 되고,
    # 지금은 간단히 경로만 직접 넣어서 쓰도록 해 둘게.
    pkl_file = "./polynomial_coefficients.pkl"
    excel_file = "polynomial_coefficients.xlsx"
    export_prm_pkl_to_excel(pkl_file, excel_file)
