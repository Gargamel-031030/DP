from pathlib import Path
import numpy as np

np.random.seed(10)
BASE_DIR = Path(__file__).resolve().parent


SCENARIO_RATIOS = {
    # Paper Scenario 1: 10% L1, 10% L2, 40% L3, 20% L4, 20% L5.
    'scenario1': ([0.5, 1.0, 2.0, 4.0, 8.0], [0.1, 0.1, 0.4, 0.2, 0.2]),
    # Paper Scenario 2: 20% L1, 20% L2, 40% L3, 10% L4, 10% L5.
    'scenario2': ([0.5, 1.0, 2.0, 4.0, 8.0], [0.2, 0.2, 0.4, 0.1, 0.1]),
    # Paper Scenario 3: 90% L1, 10% L5.
    'scenario3': ([0.5, 1.0, 2.0, 4.0, 8.0], [0.9, 0.0, 0.0, 0.0, 0.1]),
}


SCENARIO3_FIXED = {
    20: [8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 8.0],
}


def get_scenario_epsilons(scenario, N):
    if scenario not in SCENARIO_RATIOS:
        raise ValueError(f"Unknown privacy scenario: {scenario}")
    if scenario == 'scenario3' and N in SCENARIO3_FIXED:
        return list(SCENARIO3_FIXED[N])

    eps_values, ratios = SCENARIO_RATIOS[scenario]
    raw_counts = np.array(ratios, dtype=np.float64) * N
    counts = np.floor(raw_counts).astype(int)
    remainder = N - int(counts.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw_counts - counts))
        for idx in fractional_order[:remainder]:
            counts[idx] += 1

    epsilons = []
    for epsilon, count in zip(eps_values, counts):
        epsilons.extend([epsilon] * int(count))

    return epsilons


def set_epsilons(filename, N, scenario='scenario3'):
    print('=========Epsilons Info========')
    eps_path = BASE_DIR / 'epsfiles' / f'{filename}.txt'
    with open(eps_path, 'r') as rfile:
        lines = rfile.readlines()
        num_lines = len(lines)

        # 设置高斯分布的均值和标准差
        mean = num_lines / 2
        std_dev = 1.0
        # 生成符合高斯分布的随机数
        samples = np.random.normal(mean, std_dev, N)
        # 将随机数限制在0到5之间
        samples_clipped = np.clip(samples, 0.51, num_lines)
        client_safety_level = np.round(samples_clipped)
        client_safety_level = client_safety_level.astype('int')
        safety_epsilons = []
        for line in lines:
            values = line.split()
            safety_epsilons.append(float(values[1]))
        safety_epsilons = np.array(safety_epsilons)
        epsilons = [safety_epsilons[i-1] for i in client_safety_level]

    for i in range(len(epsilons)):
        if epsilons[i] == 4.0:
            epsilons[i] = 8.0
    if scenario != 'file':
        epsilons = get_scenario_epsilons(scenario, N)
    print(f'privacy_scenario:{scenario}')
    print('max_epsilons:{}, total {} values.'.format(epsilons, len(epsilons)))
    return epsilons
