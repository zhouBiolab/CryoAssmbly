import os
import sys
import re
import glob
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse


def convert_pdb_to_cif(pdb_file: str, cif_file: str):
    """
    将PDB文件转换为CIF文件
    
    参数:
        pdb_file: 输入的PDB文件路径
        cif_file: 输出的CIF文件路径
        
    返回:
        转换是否成功
    """
    try:
        from Bio.PDB import PDBParser, MMCIFIO
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(cif_file)
        
        logging.info(f"    ✅ 转换: {os.path.basename(pdb_file)} -> {os.path.basename(cif_file)}")
        return True
    except Exception as e:
        logging.warning(f"    ⚠️  PDB转CIF失败 ({os.path.basename(pdb_file)}): {e}")
        return False


class DomainComplexAssembler:
    """结构域复合物组装器 - 支持多种文件命名格式和PDB/CIF双格式"""

    def __init__(self, input_dir, output_filename=None):
        """
        初始化组装器

        Args:
            input_dir: 包含结构域文件的目录（或包含子目录的父目录）
            output_filename: 输出复合物文件名（可选，默认为assembled_complex.cif）
        """
        self.input_dir = Path(input_dir)
        self.output_filename = output_filename or "assembled_complex.cif"
        self.domain_files = []
        self.temp_cif_dir = None  # 用于存储转换的CIF文件
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _extract_chain_from_folder_name(self, folder_path):
        """
        从文件夹名称提取链ID
        支持格式:
        - chain_A, chainA, Chain_A, CHAIN_A
        - A, B, C (单字母)
        - protein_A, domain_A

        Returns:
            链ID（大写）或 None
        """
        folder_name = os.path.basename(str(folder_path))

        # 尝试匹配 chain_X, chainX 等格式
        patterns = [
            r'chain[_-]?([A-Za-z0-9]+)',
            r'protein[_-]?([A-Za-z0-9]+)',
            r'domain[_-]?([A-Za-z0-9]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, folder_name, re.IGNORECASE)
            if match:
                chain_id = match.group(1).upper()
                # 限制链ID长度（PDB格式通常是单字符）
                if len(chain_id) <= 4:
                    return chain_id

        # 如果文件夹名就是单个或少量字母数字
        if re.match(r'^[A-Za-z0-9]{1,4}$', folder_name):
            return folder_name.upper()

        return None

    def _parse_domain_number_from_filename(self, filename):
        """
        从文件名提取域号
        支持格式:
        - domain_1.pdb/cif, domain_2.pdb/cif
        - domain1.pdb/cif, domain2.pdb/cif
        - d1.pdb/cif, d2.pdb/cif
        - 1.pdb/cif, 2.pdb/cif (纯数字)
        - model_1.pdb/cif, pred_1.pdb/cif

        Returns:
            域号（整数）或 None
        """
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]

        # 尝试各种模式
        patterns = [
            r'domain[_-]?(\d+)',
            r'd[_-]?(\d+)',
            r'model[_-]?(\d+)',
            r'pred[_-]?(\d+)',
            r'component[_-]?\d+.*[_-]d[_-]?(\d+)',  # 兼容原格式
            r'^(\d+)$',  # 纯数字文件名
        ]

        for pattern in patterns:
            match = re.search(pattern, name_without_ext, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _parse_component_filename(self, filename):
        """
        解析component格式的文件名（兼容多种格式）
        支持格式:
        1. pred_链号_d_域号.pdb/cif  (新增)
        2. component_N_pred_X_M_d_D.pdb/cif (原格式)
        3. chain_X_数字_d_D.pdb/cif

        Returns:
            (chain_id, domain_num) 或 None
        """
        basename = os.path.basename(filename)

        # 格式1: pred_链号_d_域号.pdb/cif (新增支持)
        # 例如: pred_a_d_1.pdb, pred_A_d_2.cif, pred_chain1_d_3.cif
        pattern1 = r'pred_([^_]+)_d_(\d+)\.(pdb|cif)'
        match1 = re.match(pattern1, basename, re.IGNORECASE)
        if match1:
            chain_id = match1.group(1).upper()
            domain_num = int(match1.group(2))
            return (chain_id, domain_num)

        # 格式2: component_数字_pred_链ID_数字_d_域号.pdb/cif (原格式)
        pattern2 = r'component_\d+_pred_([^_]+)_\d+_d_(\d+)\.(pdb|cif)'
        match2 = re.match(pattern2, basename, re.IGNORECASE)
        if match2:
            chain_id = match2.group(1).upper()
            domain_num = int(match2.group(2))
            return (chain_id, domain_num)

        # 格式3: chain_X_数字_d_D.pdb/cif
        pattern3 = r'chain_([^_]+)_\d+_d_(\d+)\.(pdb|cif)'
        match3 = re.match(pattern3, basename, re.IGNORECASE)
        if match3:
            chain_id = match3.group(1).upper()
            domain_num = int(match3.group(2))
            return (chain_id, domain_num)

        return None

    def _read_structure_file(self, structure_file):
        """
        读取结构文件（PDB或CIF）
        
        返回:
            BioPython结构对象或None
        """
        try:
            ext = os.path.splitext(structure_file)[1].lower()
            
            if ext == '.pdb':
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True)
                return parser.get_structure('structure', structure_file)
            elif ext == '.cif':
                from Bio.PDB import MMCIFParser
                parser = MMCIFParser(QUIET=True)
                return parser.get_structure('structure', structure_file)
            else:
                logging.warning(f"不支持的文件格式: {ext}")
                return None
        except Exception as e:
            logging.warning(f"读取结构文件失败 ({structure_file}): {e}")
            return None

    def _get_residue_segments_from_file(self, structure_file):
        """
        从结构文件中读取实际的残基片段（处理断开的域）
        支持PDB和CIF格式

        Returns:
            残基片段列表 [(start1, end1), (start2, end2), ...] 或 None
        """
        try:
            structure = self._read_structure_file(structure_file)
            
            if structure is None:
                return None
            
            residue_nums = set()
            
            # 遍历结构提取残基编号
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # 跳过HETATM
                        if residue.id[0] != ' ':
                            continue
                        res_num = residue.id[1]
                        residue_nums.add(res_num)

            if not residue_nums:
                return None

            # 将残基号排序并识别连续片段
            sorted_residues = sorted(residue_nums)
            segments = []

            segment_start = sorted_residues[0]
            segment_end = sorted_residues[0]

            for i in range(1, len(sorted_residues)):
                if sorted_residues[i] == segment_end + 1:
                    # 连续，扩展当前片段
                    segment_end = sorted_residues[i]
                else:
                    # 不连续，保存当前片段并开始新片段
                    segments.append((segment_start, segment_end))
                    segment_start = sorted_residues[i]
                    segment_end = sorted_residues[i]

            # 保存最后一个片段
            segments.append((segment_start, segment_end))

            return segments

        except Exception as e:
            logging.warning(f"读取残基片段时出错 ({structure_file}): {e}")
            return None

    def _convert_pdb_files_to_cif(self, file_list):
        """
        将PDB文件转换为CIF文件
        
        参数:
            file_list: 文件路径列表
            
        返回:
            转换后的文件路径列表
        """
        # 创建临时目录存储转换的CIF文件
        if self.temp_cif_dir is None:
            self.temp_cif_dir = self.input_dir / ".temp_cif_files"
            self.temp_cif_dir.mkdir(exist_ok=True)
        
        converted_files = []
        pdb_count = 0
        
        for file_path in file_list:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.pdb':
                pdb_count += 1
                # 生成CIF文件名
                basename = os.path.basename(file_path)
                cif_filename = os.path.splitext(basename)[0] + '.cif'
                cif_path = self.temp_cif_dir / cif_filename
                
                # 转换
                if convert_pdb_to_cif(file_path, str(cif_path)):
                    converted_files.append(str(cif_path))
                else:
                    # 转换失败，保留原PDB文件
                    converted_files.append(file_path)
            else:
                # CIF文件直接使用
                converted_files.append(file_path)
        
        if pdb_count > 0:
            logging.info(f"  ✅ 转换了 {pdb_count} 个PDB文件为CIF格式")
        
        return converted_files

    def find_domain_files(self):
        """
        查找目录中的所有结构域文件（PDB和CIF）
        如果发现PDB文件，自动转换为CIF格式
        
        支持两种目录结构：
        1. 直接在input_dir下的文件（使用component格式命名）
        2. input_dir下的子目录，每个子目录代表一条链
        """
        logging.info(f"在目录中搜索结构域文件: {self.input_dir}")

        domain_info_list = []
        files_to_convert = []  # 待转换的文件列表

        # 模式1: 检查子目录结构
        subdirs = [d for d in self.input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if subdirs:
            logging.info(f"检测到 {len(subdirs)} 个子目录，尝试从子目录获取链信息...")

            for subdir in subdirs:
                chain_id = self._extract_chain_from_folder_name(subdir)

                if chain_id is None:
                    logging.warning(f"  无法从文件夹名提取链ID: {subdir.name}，跳过")
                    continue

                logging.info(f"\n  处理目录: {subdir.name} -> 链ID: {chain_id}")

                # 在子目录中查找PDB和CIF文件
                pdb_files = list(subdir.glob("*.pdb"))
                cif_files = list(subdir.glob("*.cif"))
                structure_files = pdb_files + cif_files
                
                # 如果有PDB文件，记录需要转换
                if pdb_files:
                    logging.info(f"  检测到 {len(pdb_files)} 个PDB文件，将转换为CIF格式...")
                    files_to_convert.extend([str(f) for f in pdb_files])

                # 转换PDB文件
                if pdb_files:
                    converted = self._convert_pdb_files_to_cif([str(f) for f in pdb_files])
                    # 更新文件列表，使用转换后的CIF文件
                    structure_files = [Path(f) for f in converted] + cif_files

                for structure_file in structure_files:
                    domain_num = self._parse_domain_number_from_filename(structure_file.name)

                    if domain_num is None:
                        logging.warning(f"    无法从文件名提取域号: {structure_file.name}，尝试使用默认编号")
                        # 使用文件排序作为域号
                        domain_num = len([d for d in domain_info_list if d['chain_id'] == chain_id]) + 1

                    segments = self._get_residue_segments_from_file(str(structure_file))

                    if segments:
                        min_res = segments[0][0]
                        max_res = segments[-1][1]

                        domain_info_list.append({
                            'structure_file': str(structure_file),
                            'chain_id': chain_id,
                            'domain_num': domain_num,
                            'min_residue': min_res,
                            'max_residue': max_res,
                            'segments': segments
                        })

                        segments_str = '; '.join([f"{s[0]}-{s[1]}" for s in segments])
                        file_format = os.path.splitext(structure_file)[1].upper()
                        logging.info(f"    找到: {structure_file.name} ({file_format}) -> 域{domain_num}, 残基: {segments_str}")

        # 模式2: 直接在根目录查找（使用component格式）
        else:
            logging.info("未检测到子目录，在根目录查找文件...")
            pdb_files = list(self.input_dir.glob("*.pdb"))
            cif_files = list(self.input_dir.glob("*.cif"))
            
            # 转换PDB文件
            if pdb_files:
                logging.info(f"检测到 {len(pdb_files)} 个PDB文件，将转换为CIF格式...")
                converted = self._convert_pdb_files_to_cif([str(f) for f in pdb_files])
                structure_files = [Path(f) for f in converted] + cif_files
            else:
                structure_files = cif_files

            for structure_file in structure_files:
                result = self._parse_component_filename(structure_file.name)

                if result:
                    chain_id, domain_num = result
                    segments = self._get_residue_segments_from_file(str(structure_file))

                    if segments:
                        min_res = segments[0][0]
                        max_res = segments[-1][1]

                        domain_info_list.append({
                            'structure_file': str(structure_file),
                            'chain_id': chain_id,
                            'domain_num': domain_num,
                            'min_residue': min_res,
                            'max_residue': max_res,
                            'segments': segments
                        })

                        segments_str = '; '.join([f"{s[0]}-{s[1]}" for s in segments])
                        file_format = os.path.splitext(structure_file)[1].upper()
                        logging.info(f"  找到: {structure_file.name} ({file_format}) -> 链{chain_id}, 域{domain_num}, 残基: {segments_str}")

        if not domain_info_list:
            logging.warning("未找到任何有效的结构域文件")
            return []

        logging.info(f"\n总共找到 {len(domain_info_list)} 个结构域文件（CIF格式）")
        self.domain_files = domain_info_list
        return domain_info_list

    def _group_domains_by_chain(self):
        """按链分组结构域，并将所有片段按残基位置排序"""
        logging.info("\n按链分组结构域...")

        chain_groups = defaultdict(list)

        # 先按链分组
        for domain_info in self.domain_files:
            chain_id = domain_info['chain_id']
            chain_groups[chain_id].append(domain_info)

        # 为每条链创建片段列表（每个片段记录来源域）
        chain_segments = {}

        for chain_id in chain_groups:
            all_segments = []

            for domain_info in chain_groups[chain_id]:
                segments_str = '; '.join([f"{s[0]}-{s[1]}" for s in domain_info['segments']])
                logging.info(
                    f"  链 {chain_id}, 域 {domain_info['domain_num']}: 残基片段 [{segments_str}] ({os.path.basename(domain_info['structure_file'])})")

                # 为每个片段创建记录
                for seg_start, seg_end in domain_info['segments']:
                    all_segments.append({
                        'start': seg_start,
                        'end': seg_end,
                        'domain_info': domain_info,
                        'segment': (seg_start, seg_end)
                    })

            # 按片段起始残基号排序
            all_segments.sort(key=lambda x: x['start'])
            chain_segments[chain_id] = all_segments

            logging.info(f"\n  链 {chain_id} 按残基顺序重排后的片段:")
            for seg in all_segments:
                domain_num = seg['domain_info']['domain_num']
                logging.info(f"    残基 {seg['start']}-{seg['end']} (来自域 {domain_num})")

        return chain_segments

    def _extract_atoms_from_file(self, structure_file, residue_range=None):
        """
        从结构文件中提取原子信息
        支持PDB和CIF格式

        Args:
            structure_file: 结构文件路径
            residue_range: 可选的残基范围 (start, end)

        Returns:
            原子信息列表 [(atom_name, alt_loc, residue_name, res_num, insertion_code, x, y, z, occupancy, temp_factor, element, charge), ...]
        """
        atom_info_list = []

        try:
            structure = self._read_structure_file(structure_file)
            
            if structure is None:
                return atom_info_list
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # 跳过HETATM（可根据需要调整）
                        if residue.id[0] != ' ':
                            continue
                        
                        res_num = residue.id[1]
                        insertion_code = residue.id[2] if residue.id[2] != ' ' else ''
                        
                        # 检查残基范围
                        if residue_range is not None:
                            start, end = residue_range
                            if not (start <= res_num <= end):
                                continue
                        
                        residue_name = residue.resname
                        
                        for atom in residue:
                            atom_name = atom.name
                            alt_loc = atom.altloc if hasattr(atom, 'altloc') else ''
                            coord = atom.coord
                            occupancy = atom.occupancy if hasattr(atom, 'occupancy') else 1.0
                            temp_factor = atom.bfactor if hasattr(atom, 'bfactor') else 20.0
                            element = atom.element if hasattr(atom, 'element') else ''
                            
                            atom_info_list.append({
                                'atom_name': atom_name,
                                'alt_loc': alt_loc,
                                'residue_name': residue_name,
                                'res_num': res_num,
                                'insertion_code': insertion_code,
                                'x': coord[0],
                                'y': coord[1],
                                'z': coord[2],
                                'occupancy': occupancy,
                                'temp_factor': temp_factor,
                                'element': element,
                                'charge': ''
                            })
            
        except Exception as e:
            logging.error(f"读取结构文件时出错 {structure_file}: {e}")

        return atom_info_list

    def assemble_complex(self, output_path=None):
        """
        组装复合物（CIF格式输出）

        Args:
            output_path: 输出文件路径（可选）

        Returns:
            成功返回输出文件路径，失败返回None
        """
        if not self.domain_files:
            logging.error("没有找到结构域文件，无法组装")
            return None

        if output_path is None:
            output_path = self.input_dir / self.output_filename
        else:
            output_path = Path(output_path)

        chain_segments = self._group_domains_by_chain()

        if not chain_segments:
            logging.error("无法按链分组结构域")
            return None

        logging.info(f"\n开始组装复合物...")
        logging.info(f"总链数: {len(chain_segments)}")

        try:
            # 使用BioPython创建新结构并输出为CIF
            from Bio.PDB import Structure, Model, Chain, Residue, Atom
            from Bio.PDB import MMCIFIO
            
            # 创建新的结构
            new_structure = Structure.Structure('assembled_complex')
            new_model = Model.Model(0)
            new_structure.add(new_model)
            
            global_atom_counter = 1
            
            # 按链ID排序处理
            for chain_id in sorted(chain_segments.keys()):
                segments = chain_segments[chain_id]
                
                logging.info(f"\n处理链 {chain_id}: {len(segments)} 个片段")
                
                # 创建新链
                new_chain = Chain.Chain(chain_id)
                new_model.add(new_chain)
                
                # 按残基顺序处理每个片段
                for seg_info in segments:
                    structure_file = seg_info['domain_info']['structure_file']
                    domain_num = seg_info['domain_info']['domain_num']
                    seg_start = seg_info['start']
                    seg_end = seg_info['end']
                    
                    logging.info(
                        f"  添加片段: 残基 {seg_start}-{seg_end} (来自域 {domain_num}, {os.path.basename(structure_file)})")
                    
                    # 提取该片段的原子
                    atom_info_list = self._extract_atoms_from_file(structure_file, (seg_start, seg_end))
                    
                    # 按残基组织原子
                    residues_dict = defaultdict(list)
                    for atom_info in atom_info_list:
                        res_key = (atom_info['res_num'], atom_info['insertion_code'])
                        residues_dict[res_key].append(atom_info)
                    
                    # 添加残基和原子到链
                    for res_key in sorted(residues_dict.keys()):
                        res_num, insertion_code = res_key
                        atoms = residues_dict[res_key]
                        
                        if not atoms:
                            continue
                        
                        residue_name = atoms[0]['residue_name']
                        
                        # 创建残基
                        residue_id = (' ', res_num, insertion_code if insertion_code else ' ')
                        new_residue = Residue.Residue(residue_id, residue_name, '')
                        
                        # 添加原子
                        for atom_info in atoms:
                            atom = Atom.Atom(
                                name=atom_info['atom_name'],
                                coord=[atom_info['x'], atom_info['y'], atom_info['z']],
                                bfactor=atom_info['temp_factor'],
                                occupancy=atom_info['occupancy'],
                                altloc=atom_info['alt_loc'],
                                fullname=f" {atom_info['atom_name']:<3s}",
                                serial_number=global_atom_counter,
                                element=atom_info['element']
                            )
                            new_residue.add(atom)
                            global_atom_counter += 1
                        
                        new_chain.add(new_residue)
            
            # 保存为CIF文件
            io = MMCIFIO()
            io.set_structure(new_structure)
            io.save(str(output_path))
            
            logging.info(f"\n复合物组装完成!")
            logging.info(f"输出文件: {output_path} (CIF格式)")
            logging.info(f"总原子数: {global_atom_counter - 1}")
            logging.info(f"总链数: {len(chain_segments)}")

            return str(output_path)

        except Exception as e:
            logging.error(f"组装复合物时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self):
        """运行完整的组装流程"""
        logging.info("=" * 60)
        logging.info("结构域复合物组装工具 v3.0 (CIF格式支持)")
        logging.info("=" * 60)

        # 查找结构域文件
        domain_files = self.find_domain_files()

        if not domain_files:
            logging.error("未找到有效的结构域文件")
            return None

        # 组装复合物
        output_path = self.assemble_complex()

        logging.info("=" * 60)
        return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="结构域复合物组装工具 v3.0 - 支持PDB/CIF双格式（自动转换）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的目录结构:

1. 子目录模式（推荐）:
   input_dir/
   ├── chain_A/          # 或 chainA, A, protein_A 等
   │   ├── domain_1.pdb  # 会自动转换为CIF
   │   ├── domain_2.cif  # 直接使用
   │   └── domain_3.pdb  # 会自动转换为CIF
   ├── chain_B/
   │   ├── domain_1.cif
   │   └── domain_2.cif
   └── ...

2. 直接文件模式（兼容原格式）:
   input_dir/
   ├── component_1_pred_A_1_d_1.pdb  # 会自动转换
   ├── component_2_pred_A_2_d_2.cif  # 直接使用
   └── component_3_pred_B_1_d_1.cif

支持的文件格式:
  - .pdb 文件（自动转换为CIF）
  - .cif 文件（直接使用）

支持的文件命名格式:
  - domain_1.pdb/cif, domain_2.pdb/cif
  - domain1.pdb/cif, domain2.pdb/cif
  - d1.pdb/cif, d2.pdb/cif
  - 1.pdb/cif, 2.pdb/cif
  - model_1.pdb/cif, pred_1.pdb/cif
  - component_N_pred_X_M_d_D.pdb/cif（原格式）
  - pred_X_d_Y.pdb/cif（新格式）

支持的文件夹命名（提取链ID）:
  - chain_A, chainA, Chain_A
  - protein_A, domain_A
  - A, B, C（单字母）

输出格式:
  - 默认输出为 .cif 格式
  - 所有PDB文件会自动转换为CIF格式处理

使用示例:
  # 子目录模式（混合PDB和CIF）
  python zuzhuang_refine.py /path/to/chains/

  # 指定输出文件
  python zuzhuang_refine.py /path/to/chains/ -o my_complex.cif

  # PDB文件会自动转换为CIF
  python zuzhuang_refine.py /path/to/pdb/files/
        """
    )

    parser.add_argument(
        "input_dir",
        help="包含结构域文件的目录（支持PDB和CIF格式）"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出文件名（默认: assembled_complex.cif）",
        default="assembled_complex.cif"
    )

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 - {args.input_dir}")
        return 1

    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入路径不是目录 - {args.input_dir}")
        return 1

    # 创建组装器并运行
    assembler = DomainComplexAssembler(
        input_dir=args.input_dir,
        output_filename=args.output
    )

    result = assembler.run()

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())