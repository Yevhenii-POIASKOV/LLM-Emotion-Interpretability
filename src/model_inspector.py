import torch
import numpy as np
import os
from datetime import datetime
from data_loader import get_model, get_emotional_batches

def run_model_inspection(base_path: str = "LLM-Emotion-Interpretability/data/activations") -> None:
    # --- 1. БАЗОВА ПАПКА ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(project_root, "data", "activations")  # Тут будуть збиратися всі запуски

    # --- 2. ГЕНЕРУЄМО НАЗВУ ДЛЯ НОВОГО ЗАПУСКУ ---
    # Формат: run_РРРРММДД_ГГХХ (наприклад, run_20240520_1430)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_folder_name = f"run_{timestamp}"

    # Повний шлях до конкретної папки цього запуску

    final_output_path = os.path.join(base_path, run_folder_name)

    # --- КОНФІГУРАЦІЯ МОДЕЛІ ---
    model = get_model()

    blocks_to_save = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Створюємо нову папку
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
        print(f"📂 Створено нову папку для цього запуску: {final_output_path}")

    # Сховища активацій
    activations_db = {b: [] for b in blocks_to_save}
    all_labels = []

    print(f"📊 Починаємо збір даних...")

    for batch in get_emotional_batches(batch_size=16, model=model):
        
        names_filter = [f"blocks.{b}.mlp.hook_post" for b in blocks_to_save]

        with torch.no_grad():

            _, cache = model.run_with_cache(batch.tokens, names_filter=names_filter)

        all_labels.append(batch.labels.cpu().numpy())

        for b in blocks_to_save:

            # Витягуємо активації останнього токена

            layer_acts = cache[f"blocks.{b}.mlp.hook_post"][:, -1, :].cpu().numpy()

            activations_db[b].append(layer_acts)

    print("\n💾 Запис файлів у нову папку...")

    # Зберігаємо активації
    for b in blocks_to_save:
        if activations_db[b]:
            final_matrix = np.concatenate(activations_db[b], axis=0)
            file_path = os.path.join(final_output_path, f"mlp_layer_{b}.npy")
            np.save(file_path, final_matrix)
            print(f"✅ Збережено: {f'mlp_layer_{b}.npy'}")
        else:
            print(f"⚠️ Пропущено збереження для шару {b}: не зібрано жодних активацій.")
        # Зберігаємо мітки
        if all_labels:
            np.save(os.path.join(final_output_path, "labels.npy"), np.concatenate(all_labels))
            print("✅ Збережено: labels.npy")
        else:
            print("⚠️ Пропущено збереження міток: не зібрано жодної мітки.")

    print(f"\n✨ Готово! Усі результати цього сеансу збережено тут:\n{final_output_path}")

if __name__ == "__main__":
    run_model_inspection()