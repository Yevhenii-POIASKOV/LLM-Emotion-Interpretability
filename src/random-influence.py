import torch
from transformer_lens import HookedTransformer

# 1. Завантаження моделі
# Використовуємо gpt2-small для швидкості
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 2. Підготовка промпту
prompt = "The future of artificial intelligence is"
max_tokens_to_generate = 20

# --- БАЗОВА ГЕНЕРАЦІЯ (До втручання) ---
print("\n--- ГЕНЕРАЦІЯ ДО ЗМІН (Базова) ---")
baseline_text = model.generate(prompt, max_new_tokens=max_tokens_to_generate, temperature=0.0, verbose=False)
print(baseline_text)


# 3. Створення функції-хука для рандомізації
def random_noise_hook(activation, hook):
    """
    Додає випадковий гауссівський шум до тензора активацій.
    activation: поточний тензор у точці втручання.
    hook: об'єкт хука, що містить інформацію про поточний шар та назву.
    """
    # Створюємо шум такої ж форми (shape) та на тому ж пристрої (device), що й оригінальні активації
    noise = torch.randn_like(activation)
    
    # Множимо шум на певний коефіцієнт (scale). 
    # Чим більший коефіцієнт, тим сильніше ми "ламаємо" роботу моделі.
    noise_scale = 5.0 
    
    # Змінюємо активацію: оригінал + шум
    modified_activation = activation + (noise * noise_scale)
    
    return modified_activation


# 4. Реєстрація хуків для всіх шарів
# Створюємо список кортежів: (назва_точки_втручання, функція_хук)
intervention_hooks = []
num_layers = model.cfg.n_layers

for layer in range(num_layers):
    # Додаємо хук для виходу з MLP (після функції активації)
    # Формат: [batch, position, d_mlp]
    mlp_hook_name = f"blocks.{layer}.mlp.hook_post"
    intervention_hooks.append((mlp_hook_name, random_noise_hook))
    
    # Додаємо хук для результатів голів уваги (до множення на вихідну матрицю W_O)
    # Формат: [batch, position, head_index, d_head]
    attn_hook_name = f"blocks.{layer}.attn.hook_z"
    intervention_hooks.append((attn_hook_name, random_noise_hook))


# --- ГЕНЕРАЦІЯ ПІСЛЯ ВТРУЧАННЯ (З рандомізацією) ---
print("\n--- ГЕНЕРАЦІЯ ПІСЛЯ РАНДОМНИХ ЗМІН ---")
# Використовуємо контекстний менеджер для застосування хуків під час генерації
with model.hooks(fwd_hooks=intervention_hooks):
    corrupted_text = model.generate(prompt, max_new_tokens=max_tokens_to_generate, temperature=0.0, verbose=False)
print(corrupted_text)