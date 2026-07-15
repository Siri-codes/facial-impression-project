from config import MODELS, PROMPT_VERSION, HUMAN_MEANS, N_PERMUTATIONS, SEED
from data_io import load_human_means, load_human_raw, load_model, common_stimuli
from rdm import build_rdm, compare_rdms, rdm_noise_ceiling, rsa_permutation_test
from metrics import build_evaluation_dataset, compute_self_reliability
from plots import plot_model_comparison, plot_rdm_comparison, plot_dissimilarities

def main():
    human_means = load_human_means()
    human_raw   = load_human_raw()
    models = {label: load_model(folder, f"{PROMPT_VERSION}_main.csv")
              for label, folder in MODELS.items()}

    # 1. Align BEFORE building anything
    aligned = common_stimuli(human_means, *models.values())
    human_means, models = aligned[0], dict(zip(models, aligned[1:]))

    # 2. RSA + ceiling
    human_rdm = build_rdm(human_means.drop(columns=['stimulus']).to_numpy())
    ceiling = rdm_noise_ceiling(human_raw)          # once, reused

    for label, df in models.items():
        rdm = build_rdm(df.drop(columns=['stimulus']).to_numpy())
        r, _ = compare_rdms(human_rdm, rdm)
        p = rsa_permutation_test(human_rdm, rdm, N_PERMUTATIONS, SEED)
        print(f"{label}: rho={r:.3f} (ceiling={ceiling:.3f}, p_perm={p:.4f})")
        # 3. figures...

if __name__ == "__main__":
    main()