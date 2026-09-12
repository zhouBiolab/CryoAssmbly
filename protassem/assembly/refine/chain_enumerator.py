import os
import sys
import shutil
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial

from protassem.assembly.refine.refine_energy import (
    calculate_chain_connection_energy,
    calculate_chain_connection_score,
    _calculate_tm_score
)
from protassem.assembly.refine.tr_rmsd import calculate_and_align_with_sequence


@dataclass
class DomainInfo:
    """结构域信息"""
    chain_name: str
    domain_file: str
    domain_index: int

    def __hash__(self):
        return hash((self.chain_name, self.domain_file, self.domain_index))

    def __eq__(self, other):
        return (self.chain_name == other.chain_name and
                self.domain_file == other.domain_file and
                self.domain_index == other.domain_index)


@dataclass
class AlignedDomain:
    """对齐后的结构域"""
    original_domain: DomainInfo
    target_domain: DomainInfo
    aligned_file: str
    is_original: bool
    display_name: str
    variant_index: int


@dataclass
class HomologousGroup:
    """同源结构域组"""
    group_id: int
    domains: List[DomainInfo]
    tm_scores: Dict[Tuple[str, str], float]


def convert_pdb_to_cif(pdb_file: str, cif_file: str):
    """
    将PDB文件转换为CIF文件
    
    Args:
        pdb_file: 输入的PDB文件路径
        cif_file: 输出的CIF文件路径
    """
    try:
        from Bio.PDB import PDBParser, MMCIFIO
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(cif_file)
        
        return True
    except Exception as e:
        print(f"    ⚠️  PDB转CIF失败: {e}")
        return False


def calculate_config_energy_and_score(args):
    """
    并行计算单个配置的能量和评分

    Args:
        args: (config_idx, chain_config, temp_chain_dir, ideal_distance,
               tight_tolerance, loose_min, loose_max)

    Returns:
        (config_idx, energy, score, score_details, chain_config, config_files_info)
    """
    (config_idx, chain_config, temp_chain_dir, ideal_distance,
     tight_tolerance, loose_min, loose_max) = args

    # 创建配置目录
    config_dir = os.path.join(temp_chain_dir, f"config_{config_idx}")
    os.makedirs(config_dir, exist_ok=True)

    # 复制文件到配置目录
    config_files = []
    config_files_info = []

    for pos_idx, aligned_domain in enumerate(chain_config):
        src_file = aligned_domain.aligned_file
        dst_filename = f"pos_{pos_idx}_{aligned_domain.display_name}.cif"
        dst_file = os.path.join(config_dir, dst_filename)
        shutil.copy2(src_file, dst_file)
        config_files.append(dst_file)

        # 保存文件信息用于后续写入JSON
        file_info = {
            'position': pos_idx,
            'display_name': aligned_domain.display_name,
            'variant_index': aligned_domain.variant_index,
            'is_original': aligned_domain.is_original,
            'file': dst_filename
        }
        if not aligned_domain.is_original:
            file_info['source_chain'] = aligned_domain.original_domain.chain_name
            file_info['source_file'] = os.path.basename(aligned_domain.original_domain.domain_file)

        config_files_info.append(file_info)

    # 计算能量
    energy = calculate_chain_connection_energy(config_files, ideal_distance)

    # 计算评分
    score, score_details = calculate_chain_connection_score(
        config_files,
        ideal_distance,
        tight_tolerance,
        loose_min,
        loose_max
    )

    return (config_idx, energy, score, score_details, chain_config, config_files_info)


class ChainEnumerator:
    """单链配置穷举器（支持CIF格式）"""

    def __init__(self, root_dir: str, usalign_path: str = "USalign",
                 tm_threshold: float = 0.5, ideal_distance: float = 3.8,
                 tight_tolerance: float = 1.5, loose_min: float = 5.3,
                 loose_max: float = 20.0, n_processes: int = None):
        self.root_dir = root_dir
        self.usalign_path = usalign_path
        self.tm_threshold = tm_threshold
        self.ideal_distance = ideal_distance
        self.tight_tolerance = tight_tolerance
        self.loose_min = loose_min
        self.loose_max = loose_max

        # 设置进程数
        if n_processes is None:
            self.n_processes = min(20, cpu_count())
        else:
            self.n_processes = min(n_processes, cpu_count())

        print(f"🚀 使用 {self.n_processes} 个进程进行并行计算\n")

        # 数据结构
        self.chains = {}
        self.domain_infos = {}
        self.homologous_groups = []
        self.aligned_domains = defaultdict(list)

        # 新增：完全同源链组
        self.fully_homologous_chain_groups = []

        # 工作目录
        self.work_dir = os.path.join(root_dir, "optimization_workspace")
        self.backup_dir = os.path.join(self.work_dir, "original_domains")
        self.aligned_dir = os.path.join(self.work_dir, "aligned_domains")

        self._initialize_workspace()

    def _initialize_workspace(self):
        """初始化工作空间"""
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.aligned_dir, exist_ok=True)
        print(f"✅ 工作空间创建于: {self.work_dir}\n")

    def step1_load_chains_and_backup(self):
        """加载所有链及其结构域，并备份原始文件（支持PDB自动转换为CIF）"""
        print("=" * 80)
        print("步骤1: 加载链结构并备份原始文件（支持CIF格式）")
        print("=" * 80)

        for chain_name in sorted(os.listdir(self.root_dir)):
            chain_path = os.path.join(self.root_dir, chain_name)
            if not os.path.isdir(chain_path) or chain_name == "optimization_workspace":
                continue

            domain_files = []
            files_to_rename = []  # 记录需要重命名的文件
            files_to_convert = []  # 记录需要转换的PDB文件

            # 首先扫描所有文件
            for file in sorted(os.listdir(chain_path)):
                if file.endswith('.cif'):
                    # CIF格式文件
                    if '_d_' in file:
                        # 原格式，直接添加
                        domain_files.append(os.path.join(chain_path, file))
                    elif file.startswith('domain_'):
                        # 新格式，需要重命名
                        files_to_rename.append(file)
                elif file.endswith('.pdb'):
                    # PDB格式文件，需要转换
                    if '_d_' in file:
                        files_to_convert.append((file, 'original_format'))
                    elif file.startswith('domain_'):
                        files_to_convert.append((file, 'new_format'))

            # 处理需要转换的PDB文件
            if files_to_convert:
                print(f"\n  链 {chain_name}: 检测到PDB格式文件，正在转换为CIF...")
                
                for pdb_file, format_type in files_to_convert:
                    pdb_path = os.path.join(chain_path, pdb_file)
                    
                    # 生成CIF文件名
                    if format_type == 'original_format':
                        cif_filename = pdb_file.replace('.pdb', '.cif')
                    else:
                        # domain_*.pdb 格式
                        cif_filename = pdb_file.replace('.pdb', '.cif')
                    
                    cif_path = os.path.join(chain_path, cif_filename)
                    
                    # 转换
                    if convert_pdb_to_cif(pdb_path, cif_path):
                        print(f"    ✅ {pdb_file} -> {cif_filename}")
                        
                        # 根据格式类型处理
                        if format_type == 'original_format':
                            domain_files.append(cif_path)
                        else:
                            files_to_rename.append(cif_filename)
                    else:
                        print(f"    ❌ 转换失败: {pdb_file}")

            # 如果存在 domain_*.cif 格式的文件，重命名
            if files_to_rename:
                print(f"\n  链 {chain_name}: 检测到 domain_*.cif 格式，正在转换...")

                # 按数字排序
                def extract_number(filename):
                    try:
                        # domain_1.cif -> 1
                        num_str = filename[7:-4]  # 去掉 'domain_' 和 '.cif'
                        return int(num_str)
                    except ValueError:
                        return 999999

                files_to_rename.sort(key=extract_number)

                # 重命名文件
                for idx, old_filename in enumerate(files_to_rename):
                    old_path = os.path.join(chain_path, old_filename)

                    # 提取数字
                    try:
                        domain_num = extract_number(old_filename)
                        # 生成新文件名：pred_{链号}_d_{数字}.cif
                        new_filename = f"pred_{chain_name}_d_{domain_num}.cif"
                    except:
                        # 如果无法提取数字，使用索引
                        new_filename = f"pred_{chain_name}_d_{idx}.cif"

                    new_path = os.path.join(chain_path, new_filename)

                    # 执行重命名
                    os.rename(old_path, new_path)
                    domain_files.append(new_path)

                    print(f"    ✅ {old_filename} -> {new_filename}")

            # 如果没有找到任何格式的文件，重新检查
            if not domain_files:
                for file in sorted(os.listdir(chain_path)):
                    if file.endswith('.cif') and '_d_' in file:
                        domain_files.append(os.path.join(chain_path, file))

            if domain_files:
                # 排序（按文件名）
                domain_files.sort()

                self.chains[chain_name] = domain_files

                # 备份原始文件
                backup_chain_dir = os.path.join(self.backup_dir, chain_name)
                os.makedirs(backup_chain_dir, exist_ok=True)
                for df in domain_files:
                    shutil.copy2(df, backup_chain_dir)

                # 创建DomainInfo
                for idx, df in enumerate(domain_files):
                    domain_info = DomainInfo(chain_name, df, idx)
                    self.domain_infos[df] = domain_info

                print(f"\n  链 {chain_name}: {len(domain_files)} 个结构域")
                # 显示最终的文件名
                for idx, df in enumerate(domain_files):
                    print(f"    [{idx}] {os.path.basename(df)}")

        print(f"\n✅ 共加载 {len(self.chains)} 条链")
        print(f"✅ 原始文件已备份到: {self.backup_dir}\n")

    def step2_find_homologous_domains(self):
        """寻找同源结构域组"""
        print("=" * 80)
        print("步骤2: 计算TM-score并识别同源结构域")
        print("=" * 80)

        all_domains = []
        domain_to_chain = {}

        for chain_name, domain_files in self.chains.items():
            for df in domain_files:
                all_domains.append(df)
                domain_to_chain[df] = chain_name

        # 计算TM-score矩阵
        tm_matrix = {}
        n = len(all_domains)
        print(f"需要计算 {n * (n - 1) // 2} 对TM-score...\n")

        for i in range(n):
            for j in range(i + 1, n):
                df1, df2 = all_domains[i], all_domains[j]
                chain1, chain2 = domain_to_chain[df1], domain_to_chain[df2]

                if chain1 != chain2:
                    name1 = os.path.basename(df1)
                    name2 = os.path.basename(df2)

                    rmsd, tm1, tm2 = _calculate_tm_score(df1, df2, self.usalign_path)
                    if tm1 is not None and tm2 is not None:
                        max_tm = min(tm1, tm2)
                        tm_matrix[(df1, df2)] = max_tm
                        tm_matrix[(df2, df1)] = max_tm

                        if max_tm >= self.tm_threshold:
                            print(f"  ✅ 同源对: {chain1}/{name1} ↔ {chain2}/{name2} (TM={max_tm:.3f})")

        # 使用并查集构建同源组
        parent = {df: df for df in all_domains}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for (df1, df2), tm_score in tm_matrix.items():
            if tm_score >= self.tm_threshold:
                union(df1, df2)

        # 构建同源组
        groups = defaultdict(list)
        for df in all_domains:
            root = find(df)
            groups[root].append(df)

        group_id = 0
        for root, members in groups.items():
            if len(members) > 1:
                group_tm_scores = {}
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        key = (members[i], members[j])
                        if key in tm_matrix:
                            group_tm_scores[key] = tm_matrix[key]

                group = HomologousGroup(
                    group_id=group_id,
                    domains=[self.domain_infos[df] for df in members],
                    tm_scores=group_tm_scores
                )
                self.homologous_groups.append(group)
                group_id += 1

                print(f"\n同源组 {group_id}:")
                for domain_info in group.domains:
                    print(f"  - {domain_info.chain_name}/{os.path.basename(domain_info.domain_file)}")

        print(f"\n✅ 识别出 {len(self.homologous_groups)} 个同源结构域组\n")

        # 新增：识别完全同源的链
        self._identify_fully_homologous_chains(tm_matrix)

    def _identify_fully_homologous_chains(self, tm_matrix: Dict):
        """识别完全同源的链组"""
        print("=" * 80)
        print("步骤2.5: 识别完全同源的链")
        print("=" * 80)

        chain_names = list(self.chains.keys())
        n_chains = len(chain_names)

        # 检查每对链是否完全同源
        def are_chains_fully_homologous(chain1: str, chain2: str) -> bool:
            """检查两条链是否完全同源（所有结构域都互相同源）"""
            domains1 = self.chains[chain1]
            domains2 = self.chains[chain2]

            # 结构域数量必须相同
            if len(domains1) != len(domains2):
                return False

            # 检查每个位置的结构域是否同源
            for i in range(len(domains1)):
                df1 = domains1[i]
                df2 = domains2[i]

                key1 = (df1, df2)
                key2 = (df2, df1)

                tm_score = tm_matrix.get(key1) or tm_matrix.get(key2)

                if tm_score is None or tm_score < self.tm_threshold:
                    return False

            return True

        # 使用并查集找出完全同源链组
        parent = {chain: chain for chain in chain_names}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 检查所有链对
        for i in range(n_chains):
            for j in range(i + 1, n_chains):
                chain1 = chain_names[i]
                chain2 = chain_names[j]

                if are_chains_fully_homologous(chain1, chain2):
                    union(chain1, chain2)
                    print(f"  ✅ 发现完全同源链对: {chain1} ↔ {chain2}")

        # 构建完全同源链组
        groups = defaultdict(list)
        for chain in chain_names:
            root = find(chain)
            groups[root].append(chain)

        # 只保留包含2条以上链的组
        for root, members in groups.items():
            if len(members) >= 2:
                self.fully_homologous_chain_groups.append(sorted(members))

        if self.fully_homologous_chain_groups:
            print(f"\n✅ 识别出 {len(self.fully_homologous_chain_groups)} 个完全同源链组:")
            for idx, group in enumerate(self.fully_homologous_chain_groups, 1):
                print(f"  组 {idx}: {', '.join(group)}")
        else:
            print("\n  ℹ️  未发现完全同源链组")

        print()

    def step3_generate_aligned_domains(self):
        """为每个位置预先生成所有可能的对齐版本（CIF格式）"""
        print("=" * 80)
        print("步骤3: 预生成所有对齐的结构域版本（CIF格式）")
        print("=" * 80)

        total_aligned = 0

        for chain_name, domain_files in self.chains.items():
            print(f"\n处理链: {chain_name}")
            print("-" * 60)

            for pos_idx, target_file in enumerate(domain_files):
                target_domain = self.domain_infos[target_file]
                target_basename = os.path.basename(target_file).replace('.cif', '')

                print(f"  位置 {pos_idx}: {target_basename}")

                # 变体0保存原始结构域的副本
                variant_0_filename = f"{target_basename}_0.cif"
                variant_0_filepath = os.path.join(self.aligned_dir, variant_0_filename)

                # 复制原始文件作为变体0
                shutil.copy2(target_file, variant_0_filepath)

                aligned_original = AlignedDomain(
                    original_domain=target_domain,
                    target_domain=target_domain,
                    aligned_file=variant_0_filepath,
                    is_original=True,
                    display_name=f"{target_basename}_0",
                    variant_index=0
                )
                self.aligned_domains[(chain_name, pos_idx)].append(aligned_original)
                print(f"    ✅ 变体 0 (原始): {variant_0_filename}")

                # 找同源结构域
                homolog_domains = self._get_homologous_domains(target_domain)

                if not homolog_domains:
                    print(f"    ℹ️  无同源结构域")
                    continue

                # 将同源结构域对齐到目标位置
                variant_idx = 1
                for homolog in homolog_domains:
                    if homolog.chain_name == chain_name:
                        continue

                    aligned_filename = f"{target_basename}_{variant_idx}.cif"
                    aligned_filepath = os.path.join(self.aligned_dir, aligned_filename)

                    try:
                        homolog_name = f"{homolog.chain_name}/{os.path.basename(homolog.domain_file)}"
                        print(f"    🔄 变体 {variant_idx}: 对齐 {homolog_name} -> {target_basename}")

                        calculate_and_align_with_sequence(
                            homolog.domain_file,
                            target_file,
                            aligned_filepath
                        )

                        aligned_domain = AlignedDomain(
                            original_domain=homolog,
                            target_domain=target_domain,
                            aligned_file=aligned_filepath,
                            is_original=False,
                            display_name=f"{target_basename}_{variant_idx}",
                            variant_index=variant_idx
                        )
                        self.aligned_domains[(chain_name, pos_idx)].append(aligned_domain)
                        total_aligned += 1
                        variant_idx += 1

                        print(f"    ✅ 成功生成: {aligned_filename}")

                    except Exception as e:
                        print(f"    ⚠️  对齐失败: {e}")
                        continue

        print(f"\n✅ 总共生成 {total_aligned} 个对齐变体（不含原始副本）\n")

    def _get_homologous_domains(self, domain: DomainInfo) -> List[DomainInfo]:
        """获取与给定结构域同源的所有结构域"""
        for group in self.homologous_groups:
            if domain in group.domains:
                return [d for d in group.domains if d != domain]
        return []

    def step4_enumerate_chain_configurations(self):
        """穷举每条链的所有可能配置（并行版本）"""
        print("=" * 80)
        print("步骤4: 穷举每条链的所有可能配置并并行计算能量和评分")
        print("=" * 80)

        for chain_name in sorted(self.chains.keys()):
            print(f"\n{'=' * 60}")
            print(f"处理链: {chain_name}")
            print(f"{'=' * 60}")

            # 创建临时目录
            temp_chain_dir = os.path.join(self.work_dir, f"temp_{chain_name}")
            os.makedirs(temp_chain_dir, exist_ok=True)

            # 获取该链每个位置的选项
            n_positions = len(self.chains[chain_name])
            position_options = []
            for pos_idx in range(n_positions):
                options = self.aligned_domains[(chain_name, pos_idx)]
                position_options.append(options)
                print(f"  位置 {pos_idx}: {len(options)} 个变体")

            # 计算配置总数
            total_configs = 1
            for options in position_options:
                total_configs *= len(options)

            print(f"\n  总配置数: {total_configs}")
            print(f"  使用 {self.n_processes} 个进程并行计算...\n")

            # 穷举所有配置
            all_configs = self._generate_chain_configurations(position_options)

            # 准备并行计算的参数
            calc_args = [
                (config_idx, chain_config, temp_chain_dir, self.ideal_distance,
                 self.tight_tolerance, self.loose_min, self.loose_max)
                for config_idx, chain_config in enumerate(all_configs)
            ]

            # 并行计算能量和评分
            config_results = []

            with Pool(processes=self.n_processes) as pool:
                # 使用 imap_unordered 可以实时获取结果并显示进度
                results = pool.imap_unordered(calculate_config_energy_and_score, calc_args)

                for idx, result in enumerate(results, 1):
                    (config_idx, energy, score, score_details,
                     chain_config, config_files_info) = result
                    config_results.append((config_idx, energy, score,
                                           score_details, chain_config))

                    # 保存配置信息
                    config_dir = os.path.join(temp_chain_dir, f"config_{config_idx}")
                    config_info = {
                        'config_index': config_idx,
                        'energy': energy,
                        'score': score,
                        'score_details': {
                            'tight_connections': score_details['tight_connections'],
                            'loose_connections': score_details['loose_connections'],
                            'no_score_connections': score_details['no_score_connections'],
                            'total_connections': score_details['total_connections']
                        },
                        'domains': config_files_info
                    }

                    with open(os.path.join(config_dir, "config_info.json"), 'w') as f:
                        json.dump(config_info, f, indent=2)

                    # 显示进度
                    if idx % 100 == 0 or idx == total_configs:
                        print(f"  进度: {idx}/{total_configs} ({idx / total_configs * 100:.1f}%)")

            # 排序：优先按评分降序，其次按能量升序
            config_results.sort(key=lambda x: (-x[2], x[1]))

            # 保存能量和评分报告
            report_file = os.path.join(temp_chain_dir, "energy_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"链 {chain_name} 能量与评分报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"总配置数: {total_configs}\n")
                f.write(f"结构域位置数: {n_positions}\n")
                f.write(f"并行进程数: {self.n_processes}\n\n")

                f.write("最佳前100个配置 (按评分降序，能量升序):\n")
                f.write("-" * 80 + "\n")

                for rank, (config_idx, energy, score, score_details, chain_config) in enumerate(config_results[:100],
                                                                                                1):
                    f.write(f"\n排名 {rank}:\n")
                    f.write(f"  配置编号: config_{config_idx}\n")
                    f.write(f"  评分: {score:.2f} (紧密:{score_details['tight_connections']}, "
                            f"松散:{score_details['loose_connections']}, "
                            f"无效:{score_details['no_score_connections']}, "
                            f"总计:{score_details['total_connections']})\n")
                    f.write(f"  能量: {energy:.6f}\n")
                    f.write(f"  配置详情:\n")

                    for pos_idx, aligned_domain in enumerate(chain_config):
                        if aligned_domain.is_original:
                            f.write(f"    位置 {pos_idx}: {aligned_domain.display_name} (原始)\n")
                        else:
                            f.write(f"    位置 {pos_idx}: {aligned_domain.display_name} "
                                    f"(来自 {aligned_domain.original_domain.chain_name}/"
                                    f"{os.path.basename(aligned_domain.original_domain.domain_file)})\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("所有配置能量和评分列表:\n")
                f.write("-" * 80 + "\n")

                for config_idx, energy, score, score_details, _ in config_results:
                    f.write(f"config_{config_idx}: 评分={score:.2f}, 能量={energy:.6f}\n")

            print(f"\n  ✅ 链 {chain_name} 处理完成")
            print(f"  📁 配置保存在: {temp_chain_dir}")
            print(f"  📄 能量报告: {report_file}")
            print(f"  🏆 最佳配置: config_{config_results[0][0]} "
                  f"(评分={config_results[0][2]:.2f}, 能量={config_results[0][1]:.6f})")

    def _generate_chain_configurations(self, position_options: List[List[AlignedDomain]]) -> List[List[AlignedDomain]]:
        """递归生成单条链的所有可能配置"""

        def generate(pos_idx):
            if pos_idx == len(position_options):
                return [[]]

            sub_configs = generate(pos_idx + 1)
            all_configs = []

            for option in position_options[pos_idx]:
                for sub_config in sub_configs:
                    all_configs.append([option] + sub_config)

            return all_configs

        return generate(0)

    def step5_find_global_optimum(self):
        """找到满足约束条件的全局最优配置组合（支持渐进式优化）"""
        print("=" * 80)
        print("步骤5: 寻找全局最优配置组合")
        print("=" * 80)

        # 1. 读取每条链的所有配置
        chain_configs = {}

        for chain_name in sorted(self.chains.keys()):
            temp_chain_dir = os.path.join(self.work_dir, f"temp_{chain_name}")
            configs = []

            # 读取所有配置
            for config_dir in sorted(os.listdir(temp_chain_dir)):
                if not config_dir.startswith("config_"):
                    continue

                config_path = os.path.join(temp_chain_dir, config_dir)
                info_file = os.path.join(config_path, "config_info.json")

                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        config_info = json.load(f)
                        configs.append(config_info)

            chain_configs[chain_name] = configs
            print(f"  链 {chain_name}: {len(configs)} 个配置")

        print()

        # 2. 判断是否使用渐进式优化
        if self.fully_homologous_chain_groups:
            print("  🔍 检测到完全同源链组，采用渐进式优化策略\n")
            self._progressive_optimization(chain_configs)
        else:
            print("  ℹ️  无完全同源链组，使用全局优化策略\n")
            self._global_optimization(chain_configs)

    def _progressive_optimization(self, chain_configs: Dict):
        """渐进式优化：先优化完全同源链，再优化剩余链"""

        # 已优化的链及其占用的位置
        optimized_chains = {}
        occupied_positions = set()

        # 1. 处理完全同源链组
        for group_idx, chain_group in enumerate(self.fully_homologous_chain_groups, 1):
            print(f"{'=' * 60}")
            print(f"优化完全同源链组 {group_idx}: {', '.join(chain_group)}")
            print(f"{'=' * 60}\n")

            for chain_name in chain_group:
                print(f"  优化链: {chain_name}")

                # 找到满足当前约束的最优配置
                best_config = None
                best_score = -float('inf')
                best_energy = float('inf')

                for config in chain_configs[chain_name]:
                    # 检查是否与已占用位置冲突
                    if self._config_conflicts_with_occupied(config, chain_name, occupied_positions):
                        continue

                    # 找更优的配置
                    if (config['score'] > best_score or
                            (config['score'] == best_score and config['energy'] < best_energy)):
                        best_score = config['score']
                        best_energy = config['energy']
                        best_config = config

                if best_config:
                    optimized_chains[chain_name] = best_config
                    # 更新占用位置
                    self._update_occupied_positions(best_config, chain_name, occupied_positions)
                    print(f"    ✅ 选定配置 config_{best_config['config_index']}: "
                          f"评分={best_score:.2f}, 能量={best_energy:.6f}")
                    print(f"    📍 已占用位置数: {len(occupied_positions)}\n")
                else:
                    print(f"    ❌ 未找到满足约束的配置\n")
                    return

        # 2. 处理剩余链
        remaining_chains = [c for c in sorted(self.chains.keys()) if c not in optimized_chains]

        if remaining_chains:
            print(f"{'=' * 60}")
            print(f"优化剩余链: {', '.join(remaining_chains)}")
            print(f"{'=' * 60}\n")

            # 检查剩余链是否还有完全同源组
            remaining_homologous_groups = self._find_homologous_groups_in_chains(remaining_chains)

            if remaining_homologous_groups:
                print(f"  🔍 剩余链中仍有 {len(remaining_homologous_groups)} 个完全同源组，继续渐进式优化\n")

                # 递归处理剩余的完全同源链
                optimized_remaining = set()
                for group in remaining_homologous_groups:
                    for chain_name in group:
                        if chain_name in optimized_remaining:
                            continue

                        print(f"  优化链: {chain_name}")

                        best_config = None
                        best_score = -float('inf')
                        best_energy = float('inf')

                        for config in chain_configs[chain_name]:
                            if self._config_conflicts_with_occupied(config, chain_name, occupied_positions):
                                continue

                            if (config['score'] > best_score or
                                    (config['score'] == best_score and config['energy'] < best_energy)):
                                best_score = config['score']
                                best_energy = config['energy']
                                best_config = config

                        if best_config:
                            optimized_chains[chain_name] = best_config
                            optimized_remaining.add(chain_name)
                            self._update_occupied_positions(best_config, chain_name, occupied_positions)
                            print(f"    ✅ 选定配置 config_{best_config['config_index']}: "
                                  f"评分={best_score:.2f}, 能量={best_energy:.6f}")
                            print(f"    📍 已占用位置数: {len(occupied_positions)}\n")
                        else:
                            print(f"    ❌ 未找到满足约束的配置\n")
                            return

                # 更新剩余链列表
                remaining_chains = [c for c in remaining_chains if c not in optimized_remaining]

            if remaining_chains:
                print(f"  ℹ️  剩余 {len(remaining_chains)} 条链无完全同源关系，切换到全局优化策略\n")

                # 对剩余链使用全局优化
                remaining_chain_configs = {c: chain_configs[c] for c in remaining_chains}
                best_remaining = self._optimize_remaining_chains(
                    remaining_chain_configs,
                    occupied_positions
                )

                if best_remaining:
                    optimized_chains.update(best_remaining)
                    print(f"  ✅ 剩余链优化完成\n")
                else:
                    print(f"  ❌ 剩余链优化失败\n")
                    return

        # 3. 保存结果
        if len(optimized_chains) == len(self.chains):
            total_score = sum(c['score'] for c in optimized_chains.values())
            total_energy = sum(c['energy'] for c in optimized_chains.values())

            print(f"{'=' * 80}")
            print("✅ 渐进式优化完成！")
            print(f"{'=' * 80}\n")

            self._save_optimization_result(
                optimized_chains,
                total_score,
                total_energy,
                "progressive"
            )
        else:
            print("\n❌ 优化失败：无法为所有链找到满足约束的配置")

    def _global_optimization(self, chain_configs: Dict):
        """全局优化策略"""
        chain_names = sorted(self.chains.keys())
        total_combinations = 1
        for chain_name in chain_names:
            total_combinations *= len(chain_configs[chain_name])

        print(f"  总组合数: {total_combinations:,}")

        if total_combinations > 100000:
            print(f"  ⚠️  组合数较大，计算可能需要一些时间...")

        print("  开始穷举并检查约束条件...\n")

        # 穷举并找到最优解
        best_combination = None
        best_score = -float('inf')
        best_energy = float('inf')
        valid_count = 0
        checked_count = 0

        def generate_combinations(chain_idx=0, current_combination=None,
                                  current_score=0, current_energy=0):
            nonlocal best_combination, best_score, best_energy, valid_count, checked_count

            if current_combination is None:
                current_combination = []

            if chain_idx == len(chain_names):
                checked_count += 1

                # 显示进度
                if checked_count % 10000 == 0:
                    print(f"  进度: 已检查 {checked_count:,}/{total_combinations:,} "
                          f"({checked_count / total_combinations * 100:.1f}%) | "
                          f"有效: {valid_count:,}")

                # 检查约束条件
                if self._check_global_constraints(current_combination, chain_names):
                    valid_count += 1
                    # 优先比较评分（越高越好），其次比较能量（越低越好）
                    if (current_score > best_score or
                            (current_score == best_score and current_energy < best_energy)):
                        best_score = current_score
                        best_energy = current_energy
                        best_combination = [config.copy() for config in current_combination]
                        print(f"  🎯 发现更优方案! 评分: {best_score:.2f}, 能量: {best_energy:.6f} "
                              f"(第 {valid_count} 个有效方案)")
                return

            chain_name = chain_names[chain_idx]
            for config in chain_configs[chain_name]:
                current_combination.append(config)
                generate_combinations(
                    chain_idx + 1,
                    current_combination,
                    current_score + config['score'],
                    current_energy + config['energy']
                )
                current_combination.pop()

        # 开始穷举
        generate_combinations()

        print(f"\n  ✅ 穷举完成!")
        print(f"  总组合数: {checked_count:,}")
        print(f"  有效方案数: {valid_count:,}")
        if checked_count > 0:
            print(f"  约束满足率: {valid_count / checked_count * 100:.2f}%")

        if best_combination:
            print(f"  🏆 最优评分: {best_score:.2f}")
            print(f"  🏆 最优能量: {best_energy:.6f}\n")

            # 转换为字典格式
            optimized_chains = {}
            for i, chain_name in enumerate(chain_names):
                optimized_chains[chain_name] = best_combination[i]

            self._save_optimization_result(
                optimized_chains,
                best_score,
                best_energy,
                "global"
            )
        else:
            print("\n❌ 未找到满足约束条件的方案!")

    def _find_homologous_groups_in_chains(self, chain_list: List[str]) -> List[List[str]]:
        """在给定的链列表中寻找完全同源组"""
        groups = []
        processed = set()

        for i, chain1 in enumerate(chain_list):
            if chain1 in processed:
                continue

            current_group = [chain1]

            for chain2 in chain_list[i + 1:]:
                if chain2 in processed:
                    continue

                # 检查chain2是否与current_group中所有链都完全同源
                is_homologous_with_all = True
                for chain in current_group:
                    if not self._are_chains_fully_homologous(chain, chain2):
                        is_homologous_with_all = False
                        break

                if is_homologous_with_all:
                    current_group.append(chain2)
                    processed.add(chain2)

            if len(current_group) >= 2:
                groups.append(current_group)
                processed.add(chain1)

        return groups

    def _are_chains_fully_homologous(self, chain1: str, chain2: str) -> bool:
        """检查两条链是否完全同源"""
        for group in self.fully_homologous_chain_groups:
            if chain1 in group and chain2 in group:
                return True
        return False

    def _config_conflicts_with_occupied(self, config: Dict, chain_name: str,
                                        occupied_positions: Set[Tuple[str, int]]) -> bool:
        """检查配置是否与已占用位置冲突"""
        for domain in config['domains']:
            if domain['is_original']:
                key = (chain_name, domain['position'])
                if key in occupied_positions:
                    return True
            else:
                source_chain = domain['source_chain']
                source_file = domain['source_file']
                source_position = self._find_source_position(source_chain, source_file)

                if source_position != -1:
                    key = (source_chain, source_position)
                    if key in occupied_positions:
                        return True

        return False

    def _update_occupied_positions(self, config: Dict, chain_name: str,
                                   occupied_positions: Set[Tuple[str, int]]):
        """更新已占用位置集合"""
        for domain in config['domains']:
            if domain['is_original']:
                key = (chain_name, domain['position'])
                occupied_positions.add(key)
            else:
                source_chain = domain['source_chain']
                source_file = domain['source_file']
                source_position = self._find_source_position(source_chain, source_file)

                if source_position != -1:
                    key = (source_chain, source_position)
                    occupied_positions.add(key)

    def _optimize_remaining_chains(self, remaining_chain_configs: Dict,
                                   occupied_positions: Set[Tuple[str, int]]) -> Dict:
        """对剩余链进行全局优化"""
        if not remaining_chain_configs:
            return {}

        chain_names = sorted(remaining_chain_configs.keys())
        best_combination = None
        best_score = -float('inf')
        best_energy = float('inf')

        def generate_combinations(chain_idx=0, current_combination=None,
                                  current_score=0, current_energy=0):
            nonlocal best_combination, best_score, best_energy

            if current_combination is None:
                current_combination = []

            if chain_idx == len(chain_names):
                # 检查约束条件
                temp_occupied = occupied_positions.copy()
                valid = True

                for i, config in enumerate(current_combination):
                    chain_name = chain_names[i]
                    if self._config_conflicts_with_occupied(config, chain_name, temp_occupied):
                        valid = False
                        break
                    self._update_occupied_positions(config, chain_name, temp_occupied)

                if valid:
                    if (current_score > best_score or
                            (current_score == best_score and current_energy < best_energy)):
                        best_score = current_score
                        best_energy = current_energy
                        best_combination = [config.copy() for config in current_combination]
                return

            chain_name = chain_names[chain_idx]
            for config in remaining_chain_configs[chain_name]:
                current_combination.append(config)
                generate_combinations(
                    chain_idx + 1,
                    current_combination,
                    current_score + config['score'],
                    current_energy + config['energy']
                )
                current_combination.pop()

        generate_combinations()

        if best_combination:
            result = {}
            for i, chain_name in enumerate(chain_names):
                result[chain_name] = best_combination[i]
            return result

        return {}

    def _check_global_constraints(self, combination, chain_names):
        """检查约束条件：每个原始结构域位置只能被一条链占据"""
        domain_usage = {}

        for i, config in enumerate(combination):
            chain_name = chain_names[i]

            for domain in config['domains']:
                position = domain['position']
                is_original = domain['is_original']

                if is_original:
                    key = (chain_name, position)
                    if key in domain_usage:
                        return False
                    domain_usage[key] = chain_name
                else:
                    source_chain = domain['source_chain']
                    source_file = domain['source_file']
                    source_position = self._find_source_position(source_chain, source_file)

                    if source_position == -1:
                        continue

                    key = (source_chain, source_position)
                    if key in domain_usage:
                        return False
                    domain_usage[key] = chain_name

        return True

    def _find_source_position(self, source_chain, source_file):
        """从source_file中找到对应的位置索引"""
        if source_chain not in self.chains:
            return -1

        for idx, domain_file in enumerate(self.chains[source_chain]):
            if os.path.basename(domain_file) == source_file:
                return idx

        return -1

    def _save_optimization_result(self, optimized_chains: Dict, total_score: float,
                                  total_energy: float, method: str):
        """保存优化结果"""
        result_file = os.path.join(self.work_dir, "global_optimal_solution.json")

        result_data = {
            'optimization_method': method,
            'total_score': total_score,
            'total_energy': total_energy,
            'chains': optimized_chains
        }

        if method == "progressive":
            result_data['fully_homologous_chain_groups'] = self.fully_homologous_chain_groups

        print(f"\n{'=' * 80}")
        print("🏆 最优方案")
        print(f"{'=' * 80}\n")
        print(f"优化方法: {'渐进式优化' if method == 'progressive' else '全局优化'}")
        print(f"总评分: {total_score:.2f}")
        print(f"总能量: {total_energy:.6f}\n")

        for chain_name in sorted(optimized_chains.keys()):
            config = optimized_chains[chain_name]
            score_details = config['score_details']
            print(f"链 {chain_name}:")
            print(f"  配置编号: config_{config['config_index']}")
            print(f"  评分: {config['score']:.2f} (紧密:{score_details['tight_connections']}, "
                  f"松散:{score_details['loose_connections']}, "
                  f"无效:{score_details['no_score_connections']}, "
                  f"总计:{score_details['total_connections']})")
            print(f"  能量: {config['energy']:.6f}")
            print(f"  配置详情:")

            for domain in config['domains']:
                if domain['is_original']:
                    print(f"    位置 {domain['position']}: {domain['display_name']} (原始)")
                else:
                    print(f"    位置 {domain['position']}: {domain['display_name']} "
                          f"(来自 {domain['source_chain']}/{domain['source_file']})")
            print()

        # 保存JSON
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        # 保存文本报告
        report_file = os.path.join(self.work_dir, "global_optimal_solution.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("最优方案报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"优化方法: {'渐进式优化' if method == 'progressive' else '全局优化'}\n")
            f.write(f"总评分: {total_score:.2f}\n")
            f.write(f"总能量: {total_energy:.6f}\n\n")

            if method == "progressive" and self.fully_homologous_chain_groups:
                f.write("完全同源链组:\n")
                for idx, group in enumerate(self.fully_homologous_chain_groups, 1):
                    f.write(f"  组 {idx}: {', '.join(group)}\n")
                f.write("\n")

            for chain_name in sorted(optimized_chains.keys()):
                config = optimized_chains[chain_name]
                score_details = config['score_details']
                f.write(f"{'=' * 60}\n")
                f.write(f"链 {chain_name}\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"配置编号: config_{config['config_index']}\n")
                f.write(f"评分: {config['score']:.2f} (紧密:{score_details['tight_connections']}, "
                        f"松散:{score_details['loose_connections']}, "
                        f"无效:{score_details['no_score_connections']}, "
                        f"总计:{score_details['total_connections']})\n")
                f.write(f"能量: {config['energy']:.6f}\n")
                f.write(f"配置目录: temp_{chain_name}/config_{config['config_index']}\n\n")
                f.write("配置详情:\n")

                for domain in config['domains']:
                    if domain['is_original']:
                        f.write(f"  位置 {domain['position']}: {domain['display_name']} (原始)\n")
                    else:
                        f.write(f"  位置 {domain['position']}: {domain['display_name']} "
                                f"(来自 {domain['source_chain']}/{domain['source_file']})\n")
                f.write("\n")

        print(f"✅ 结果已保存:")
        print(f"  📄 JSON: {result_file}")
        print(f"  📄 TXT:  {report_file}")

    def step6_extract_optimal_structures(self):
        """提取最优方案的结构文件到final_results文件夹，并组装成复合物（CIF格式）"""
        print("=" * 80)
        print("步骤6: 提取最优方案的结构文件并组装复合物（CIF格式）")
        print("=" * 80)

        # 读取最优方案
        solution_file = os.path.join(self.work_dir, "global_optimal_solution.json")

        if not os.path.exists(solution_file):
            print("❌ 未找到最优方案文件，请先运行优化流程")
            return

        with open(solution_file, 'r', encoding='utf-8') as f:
            solution = json.load(f)

        # 创建final_results文件夹
        final_results_dir = os.path.join(self.work_dir, "final_results")
        if os.path.exists(final_results_dir):
            shutil.rmtree(final_results_dir)
        os.makedirs(final_results_dir)

        print(f"✅ 创建输出目录: {final_results_dir}\n")

        total_files = 0

        # 遍历每条链的最优配置
        for chain_name in sorted(solution['chains'].keys()):
            config = solution['chains'][chain_name]
            config_index = config['config_index']

            print(f"处理链 {chain_name}:")
            print(f"  配置编号: config_{config_index}")

            # 找到配置目录
            config_dir = os.path.join(self.work_dir, f"temp_{chain_name}", f"config_{config_index}")

            if not os.path.exists(config_dir):
                print(f"  ⚠️  警告: 配置目录不存在: {config_dir}")
                continue

            # 获取该目录下所有CIF文件
            cif_files = [f for f in os.listdir(config_dir) if f.endswith('.cif')]

            if not cif_files:
                print(f"  ⚠️  警告: 配置目录中没有CIF文件")
                continue

            # 复制并重命名文件
            for cif_file in sorted(cif_files):
                src_file = os.path.join(config_dir, cif_file)

                # 去掉前6位（"pos_X_"）和后2位（"_Y"）
                # 例如: pos_0_AF-Q9UBB4-F1-model_v4_d_0_0.cif
                # 变成: AF-Q9UBB4-F1-model_v4_d_0.cif

                # 分析文件名结构
                if cif_file.startswith("pos_"):
                    # 找到第二个下划线的位置（去掉"pos_X_"）
                    parts = cif_file.split('_', 2)
                    if len(parts) >= 3:
                        # parts[2] 是剩余部分
                        remaining = parts[2]

                        # 去掉.cif后缀
                        if remaining.endswith('.cif'):
                            remaining = remaining[:-4]

                        # 去掉最后的"_Y"部分（变体索引）
                        name_parts = remaining.rsplit('_', 1)
                        if len(name_parts) >= 2:
                            new_name = name_parts[0]
                        else:
                            new_name = remaining

                        new_filename = f"{new_name}.cif"
                    else:
                        # 如果格式不符合预期，保留原文件名
                        new_filename = cif_file
                else:
                    # 如果格式不符合预期，保留原文件名
                    new_filename = cif_file

                dst_file = os.path.join(final_results_dir, new_filename)

                # 如果文件名冲突，添加链名区分
                if os.path.exists(dst_file):
                    base_name = new_filename[:-4]  # 去掉.cif
                    new_filename = f"{chain_name}_{base_name}.cif"
                    dst_file = os.path.join(final_results_dir, new_filename)
                    print(f"  ⚠️  文件名冲突，添加链名前缀: {new_filename}")

                shutil.copy2(src_file, dst_file)

                print(f"  ✅ {cif_file} -> {new_filename}")
                total_files += 1

            print()

        print(f"{'=' * 80}")
        print(f"✅ 提取完成！共复制 {total_files} 个结构文件（CIF格式）")
        print(f"📁 输出目录: {final_results_dir}")
        print(f"{'=' * 80}\n")

        # 生成一个README文件说明
        readme_file = os.path.join(final_results_dir, "README.txt")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("最优方案结构文件说明（CIF格式）\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"优化方法: {solution['optimization_method']}\n")
            f.write(f"总评分: {solution['total_score']:.2f}\n")
            f.write(f"总能量: {solution['total_energy']:.6f}\n\n")

            f.write("文件格式: mmCIF (.cif)\n\n")

            f.write("文件命名规则:\n")
            f.write("  格式: {结构域名}.cif\n")
            f.write("  说明: 已去除位置前缀(pos_X_)和变体后缀(_Y)\n")
            f.write("  注意: 如有重名文件会添加链名前缀以区分\n\n")

            f.write("各链配置详情:\n")
            f.write("-" * 80 + "\n\n")

            for chain_name in sorted(solution['chains'].keys()):
                config = solution['chains'][chain_name]
                score_details = config['score_details']

                f.write(f"链 {chain_name}:\n")
                f.write(f"  配置编号: config_{config['config_index']}\n")
                f.write(f"  评分: {config['score']:.2f} ")
                f.write(f"(紧密:{score_details['tight_connections']}, ")
                f.write(f"松散:{score_details['loose_connections']}, ")
                f.write(f"无效:{score_details['no_score_connections']}, ")
                f.write(f"总计:{score_details['total_connections']})\n")
                f.write(f"  能量: {config['energy']:.6f}\n")
                f.write(f"  结构域数量: {len(config['domains'])}\n\n")

            f.write("=" * 80 + "\n")

        print(f"📄 已生成说明文件: {readme_file}\n")

        # ====== 新增：调用 DomainComplexAssembler 组装复合物 ======
        print("=" * 80)
        print("步骤6.5: 组装最终复合物结构（CIF格式）")
        print("=" * 80)

        try:
            from protassem.assembly.refine.complex_assembler import DomainComplexAssembler

            assembler = DomainComplexAssembler(
                input_dir=final_results_dir,
                output_filename="final.cif"  # 修改为CIF格式
            )

            # 查找结构域文件
            domain_files = assembler.find_domain_files()

            if not domain_files:
                print("⚠️  警告: 未找到有效的结构域文件，跳过组装步骤")
            else:
                # 组装复合物
                output_path = os.path.join(final_results_dir, "final.cif")
                result = assembler.assemble_complex(output_path=output_path)

                if result:
                    print(f"\n✅ 复合物组装成功！")
                    print(f"📁 最终复合物文件（CIF格式）: {result}")
                else:
                    print("\n⚠️  复合物组装失败")

        except ImportError as e:
            print(f"⚠️  警告: 无法导入 DomainComplexAssembler 模块")
            print(f"   请确保 zuzhuang_refine.py 在相同目录下")
            print(f"   错误信息: {e}")
        except Exception as e:
            print(f"⚠️  组装复合物时发生错误: {e}")
            import traceback
            traceback.print_exc()

        print()

    def run_enumeration(self):
        """运行完整的穷举流程"""
        print("\n" + "🔬" * 40)
        print("蛋白质复合物结构域优化（智能优化版 - 支持CIF格式）")
        print("🔬" * 40 + "\n")

        try:
            self.step1_load_chains_and_backup()
            self.step2_find_homologous_domains()

            if not self.homologous_groups:
                print("⚠️ 未发现同源结构域，无需优化")
                return

            self.step3_generate_aligned_domains()
            self.step4_enumerate_chain_configurations()
            self.step5_find_global_optimum()
            self.step6_extract_optimal_structures()  # 新增步骤

            print("\n" + "=" * 80)
            print("✅ 全部流程完成！")
            print("=" * 80)
            print(f"\n📁 所有结果保存在: {self.work_dir}")
            print(f"\n🎯 查看最优方案:")
            print(f"  - {os.path.join(self.work_dir, 'global_optimal_solution.json')}")
            print(f"  - {os.path.join(self.work_dir, 'global_optimal_solution.txt')}")
            print(f"  - {os.path.join(self.work_dir, 'final_results')} (最优结构文件，CIF格式)")

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python chain_enumerator_cif.py <root_dir> [options]")
        print("\n选项:")
        print("  --tm-threshold <float>   TM-score阈值 (默认: 0.85)")
        print("  --ideal-distance <float> 理想CA-CA距离 (默认: 3.8)")
        print("  --tight-tolerance <float> 紧密连接容差 (默认: 1.5)")
        print("  --loose-min <float>      松散连接最小距离 (默认: 5.3)")
        print("  --loose-max <float>      松散连接最大距离 (默认: 20.0)")
        print("  --usalign <path>         USalign路径")
        print("  --n-processes <int>      并行进程数 (默认: 20，最大为CPU核心数)")
        print("\n示例:")
        print("  python chain_enumerator_cif.py ./protein_complex --tm-threshold 0.7 --n-processes 20")
        print("\n注意:")
        print("  - 程序支持.pdb和.cif格式输入文件")
        print("  - .pdb文件将自动转换为.cif格式")
        print("  - 所有输出文件均为.cif格式")
        sys.exit(1)

    root_dir = sys.argv[1]

    # 解析参数
    tm_threshold = 0.75
    ideal_distance = 3.8
    tight_tolerance = 1.5
    loose_min = 5.3
    loose_max = 10
    usalign_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "core", "USalign")
    n_processes = 20

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--tm-threshold" and i + 1 < len(sys.argv):
            tm_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--ideal-distance" and i + 1 < len(sys.argv):
            ideal_distance = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--tight-tolerance" and i + 1 < len(sys.argv):
            tight_tolerance = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--loose-min" and i + 1 < len(sys.argv):
            loose_min = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--loose-max" and i + 1 < len(sys.argv):
            loose_max = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--usalign" and i + 1 < len(sys.argv):
            usalign_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--n-processes" and i + 1 < len(sys.argv):
            n_processes = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    # 运行穷举
    enumerator = ChainEnumerator(
        root_dir=root_dir,
        usalign_path=usalign_path,
        tm_threshold=tm_threshold,
        ideal_distance=ideal_distance,
        tight_tolerance=tight_tolerance,
        loose_min=loose_min,
        loose_max=loose_max,
        n_processes=n_processes
    )

    enumerator.run_enumeration()


if __name__ == "__main__":
    main()