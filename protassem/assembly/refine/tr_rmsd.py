"""
基于序列比对计算两个结构文件（PDB/CIF）之间的 RMSD，
并将第二个结构对齐到第一个结构，输出对齐后的新文件。

支持格式：PDB (.pdb) 和 mmCIF (.cif)
"""

import sys
import os
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO, Superimposer
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# 氨基酸三字母码到单字母码的映射
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def get_parser_for_file(filepath):
    """
    根据文件扩展名返回合适的parser
    
    参数:
        filepath: 结构文件路径
        
    返回:
        parser对象和文件格式('pdb'或'cif')
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.pdb':
        return PDBParser(QUIET=True), 'pdb'
    elif ext == '.cif':
        return MMCIFParser(QUIET=True), 'cif'
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持.pdb和.cif")


def get_io_for_format(output_file):
    """
    根据输出文件扩展名返回合适的IO对象
    
    参数:
        output_file: 输出文件路径
        
    返回:
        IO对象
    """
    ext = os.path.splitext(output_file)[1].lower()
    
    if ext == '.pdb':
        return PDBIO()
    elif ext == '.cif':
        return MMCIFIO()
    else:
        # 默认使用PDBIO（向后兼容）
        return PDBIO()


def get_sequence_and_ca(structure):
    """
    返回序列字符串和 {position: (resid, CA_atom)} 字典
    
    参数:
        structure: BioPython结构对象
        
    返回:
        (sequence, ca_info): 序列字符串和CA原子信息字典
    """
    sequence = ""
    ca_info = {}
    position = 0
    
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname() in AA_MAP and "CA" in res:
                    aa = AA_MAP[res.get_resname()]
                    sequence += aa
                    resid = res.get_id()[1]
                    ca_info[position] = (resid, res["CA"])
                    position += 1
    
    return sequence, ca_info


def find_best_alignment(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-0.5):
    """
    使用全局比对找到最佳序列比对
    
    参数:
        seq1: 第一个序列
        seq2: 第二个序列
        match_score: 匹配得分
        mismatch_penalty: 错配罚分
        gap_penalty: gap罚分
        
    返回:
        alignment对象
    """
    alignments = pairwise2.align.globalms(
        seq1, seq2,
        match_score, mismatch_penalty, gap_penalty, gap_penalty,
        one_alignment_only=True
    )
    
    if not alignments:
        raise ValueError("无法找到有效的序列比对！")
    
    return alignments[0]


def get_aligned_residues(alignment, ca_info1, ca_info2):
    """
    根据序列比对结果获取对应的残基对
    
    参数:
        alignment: 序列比对结果
        ca_info1: 第一个结构的CA信息
        ca_info2: 第二个结构的CA信息
        
    返回:
        (atoms1, atoms2, matched_pairs): 匹配的原子对和残基对信息
    """
    aligned_seq1, aligned_seq2, score, begin, end = alignment
    
    atoms1, atoms2 = [], []
    pos1, pos2 = 0, 0
    matched_pairs = []
    
    for i in range(len(aligned_seq1)):
        char1, char2 = aligned_seq1[i], aligned_seq2[i]
        
        if char1 != '-' and char2 != '-':  # 匹配位置
            if char1 == char2:  # 相同氨基酸
                if pos1 in ca_info1 and pos2 in ca_info2:
                    resid1, atom1 = ca_info1[pos1]
                    resid2, atom2 = ca_info2[pos2]
                    atoms1.append(atom1)
                    atoms2.append(atom2)
                    matched_pairs.append((resid1, resid2, char1))
        
        if char1 != '-':
            pos1 += 1
        if char2 != '-':
            pos2 += 1
    
    return atoms1, atoms2, matched_pairs


def calculate_and_align_with_sequence(structure_file1, structure_file2, output_file="aligned.pdb", 
                                    match_score=2, mismatch_penalty=-1, gap_penalty=-0.5):
    """
    基于序列比对计算RMSD并对齐两个结构
    支持PDB和CIF格式
    
    参数:
        structure_file1: 参考结构文件路径（.pdb或.cif）
        structure_file2: 待对齐结构文件路径（.pdb或.cif）
        output_file: 输出文件路径（.pdb或.cif，默认.pdb）
        match_score: 序列匹配得分
        mismatch_penalty: 序列错配罚分
        gap_penalty: gap罚分
    """
    # 自动选择合适的parser
    parser1, format1 = get_parser_for_file(structure_file1)
    parser2, format2 = get_parser_for_file(structure_file2)
    
    print(f"读取文件1: {structure_file1} (格式: {format1.upper()})")
    print(f"读取文件2: {structure_file2} (格式: {format2.upper()})")

    # 解析两个结构
    structure1 = parser1.get_structure("ref", structure_file1)
    structure2 = parser2.get_structure("mob", structure_file2)

    # 获取序列和Cα原子信息
    seq1, ca_info1 = get_sequence_and_ca(structure1)
    seq2, ca_info2 = get_sequence_and_ca(structure2)
    
    print(f"结构1序列长度: {len(seq1)}")
    print(f"结构2序列长度: {len(seq2)}")
    print(f"结构1序列: {seq1}")
    print(f"结构2序列: {seq2}")
    print()

    # 进行序列比对
    print("进行序列比对...")
    alignment = find_best_alignment(seq1, seq2, match_score, mismatch_penalty, gap_penalty)
    
    print("比对结果:")
    print(format_alignment(*alignment))
    
    # 获取对应的残基对
    atoms1, atoms2, matched_pairs = get_aligned_residues(alignment, ca_info1, ca_info2)
    
    if len(atoms1) == 0:
        raise ValueError("没有找到匹配的残基对，无法对齐！")
    
    print(f"找到 {len(atoms1)} 个匹配的残基对:")
    for i, (resid1, resid2, aa) in enumerate(matched_pairs[:10]):  # 只显示前10个
        print(f"  {resid1} ({aa}) <-> {resid2} ({aa})")
    if len(matched_pairs) > 10:
        print(f"  ... 还有 {len(matched_pairs)-10} 个匹配对")
    print()

    # 对齐并计算 RMSD
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(structure2.get_atoms())

    print(f"基于序列比对使用 {len(atoms1)} 个残基进行结构叠合")
    print(f"序列相似性得分: {alignment[2]:.1f}")
    print(f"RMSD = {sup.rms:.3f} Å")

    # 根据输出文件格式选择IO
    io = get_io_for_format(output_file)
    io.set_structure(structure2)
    io.save(output_file)
    
    output_format = os.path.splitext(output_file)[1].upper()
    print(f"对齐后的结构已保存到: {output_file} (格式: {output_format})")


def calculate_and_align_simple(structure_file1, structure_file2, output_file="aligned.pdb"):
    """
    原始的简单方法（基于残基编号）
    支持PDB和CIF格式
    
    参数:
        structure_file1: 参考结构文件路径（.pdb或.cif）
        structure_file2: 待对齐结构文件路径（.pdb或.cif）
        output_file: 输出文件路径（.pdb或.cif，默认.pdb）
    """
    # 自动选择合适的parser
    parser1, format1 = get_parser_for_file(structure_file1)
    parser2, format2 = get_parser_for_file(structure_file2)
    
    print("使用简单的残基编号匹配方法...")
    print(f"读取文件1: {structure_file1} (格式: {format1.upper()})")
    print(f"读取文件2: {structure_file2} (格式: {format2.upper()})")
    
    structure1 = parser1.get_structure("ref", structure_file1)
    structure2 = parser2.get_structure("mob", structure_file2)
    
    # 获取所有CA原子
    atoms1, atoms2 = [], []
    
    for model1, model2 in zip(structure1, structure2):
        for chain1, chain2 in zip(model1, model2):
            for res1, res2 in zip(chain1, chain2):
                if res1.get_resname() in AA_MAP and "CA" in res1:
                    if res2.get_resname() in AA_MAP and "CA" in res2:
                        atoms1.append(res1["CA"])
                        atoms2.append(res2["CA"])
    
    if len(atoms1) == 0:
        raise ValueError("没有找到匹配的残基对，无法对齐！")
    
    print(f"找到 {len(atoms1)} 个残基对")
    
    # 对齐并计算RMSD
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(structure2.get_atoms())
    
    print(f"使用 {len(atoms1)} 个残基进行结构叠合")
    print(f"RMSD = {sup.rms:.3f} Å")
    
    # 根据输出文件格式选择IO
    io = get_io_for_format(output_file)
    io.set_structure(structure2)
    io.save(output_file)
    
    output_format = os.path.splitext(output_file)[1].upper()
    print(f"对齐后的结构已保存到: {output_file} (格式: {output_format})")


def calculate_and_align_by_resid(structure_file1, structure_file2, output_file="aligned.pdb"):
    """
    按残基编号直接配对CA原子进行对齐（适用于域是链的子集的情况）。

    域的残基编号是链残基编号的子集，直接按resSeq匹配，
    不需要序列比对，保证RMSD=0。

    参数:
        structure_file1: 参考结构文件（已拟合的链，.pdb或.cif）
        structure_file2: 待对齐结构文件（原始域，.pdb或.cif）
        output_file: 输出文件路径
    """
    parser1, format1 = get_parser_for_file(structure_file1)
    parser2, format2 = get_parser_for_file(structure_file2)

    print(f"使用残基编号匹配模式 (--resid)")
    print(f"读取文件1(参考): {structure_file1} (格式: {format1.upper()})")
    print(f"读取文件2(待对齐): {structure_file2} (格式: {format2.upper()})")

    structure1 = parser1.get_structure("ref", structure_file1)
    structure2 = parser2.get_structure("mob", structure_file2)

    # 提取参考结构的 {(chainID, resSeq): CA_atom}
    ca_dict1 = {}
    for model in structure1:
        for chain in model:
            for res in chain:
                if res.get_resname() in AA_MAP and "CA" in res:
                    key = (chain.get_id(), res.get_id()[1])
                    ca_dict1[key] = res["CA"]

    # 提取待对齐结构的 {(chainID, resSeq): CA_atom}
    ca_dict2 = {}
    for model in structure2:
        for chain in model:
            for res in chain:
                if res.get_resname() in AA_MAP and "CA" in res:
                    key = (chain.get_id(), res.get_id()[1])
                    ca_dict2[key] = res["CA"]

    print(f"参考结构CA原子数: {len(ca_dict1)}")
    print(f"待对齐结构CA原子数: {len(ca_dict2)}")

    # 按残基编号取交集配对（忽略chainID，只按resSeq匹配）
    resseq_set1 = {resseq for (_, resseq) in ca_dict1}
    resseq_set2 = {resseq for (_, resseq) in ca_dict2}
    common_resseqs = sorted(resseq_set1 & resseq_set2)

    if not common_resseqs:
        raise ValueError("没有找到共同的残基编号，无法对齐！")

    # 构建配对的原子列表
    atoms1, atoms2 = [], []
    # 对参考结构，按resSeq建索引（取第一条链的）
    resseq_to_atom1 = {}
    for (cid, resseq), atom in ca_dict1.items():
        if resseq not in resseq_to_atom1:
            resseq_to_atom1[resseq] = atom

    resseq_to_atom2 = {}
    for (cid, resseq), atom in ca_dict2.items():
        if resseq not in resseq_to_atom2:
            resseq_to_atom2[resseq] = atom

    for resseq in common_resseqs:
        atoms1.append(resseq_to_atom1[resseq])
        atoms2.append(resseq_to_atom2[resseq])

    print(f"匹配的残基数: {len(atoms1)}")
    print(f"残基编号范围: {common_resseqs[0]} - {common_resseqs[-1]}")

    # 对齐
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(structure2.get_atoms())

    print(f"RMSD = {sup.rms:.3f} Å")

    # 保存
    io = get_io_for_format(output_file)
    io.set_structure(structure2)
    io.save(output_file)

    output_format = os.path.splitext(output_file)[1].upper()
    print(f"对齐后的结构已保存到: {output_file} (格式: {output_format})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python tr_rmsd.py structure1 structure2 [output] [--simple] [--resid]")
        print()
        print("参数:")
        print("  structure1: 参考结构文件 (.pdb 或 .cif)")
        print("  structure2: 待对齐结构文件 (.pdb 或 .cif)")
        print("  output:     输出文件路径 (.pdb 或 .cif，默认: aligned.pdb)")
        print("  --simple:   使用简单的残基编号匹配方法（可选）")
        print("  --resid:    按残基编号直接配对（域是链子集时使用，保证RMSD=0）")
        print()
        print("示例:")
        print("  python tr_rmsd.py model1.pdb model2.pdb")
        print("  python tr_rmsd.py model1.cif model2.cif aligned.cif")
        print("  python tr_rmsd.py model1.pdb model2.cif aligned.pdb")
        print("  python tr_rmsd.py model1.pdb model2.pdb aligned.pdb --simple")
        print("  python tr_rmsd.py fitted_chain.pdb domain.pdb aligned.pdb --resid")
        sys.exit(1)

    structure_file1 = sys.argv[1]
    structure_file2 = sys.argv[2]

    # 解析参数
    use_simple = "--simple" in sys.argv
    use_resid = "--resid" in sys.argv
    if use_simple:
        sys.argv.remove("--simple")
    if use_resid:
        sys.argv.remove("--resid")

    output_file = sys.argv[3] if len(sys.argv) > 3 else "aligned.pdb"

    try:
        if use_resid:
            calculate_and_align_by_resid(structure_file1, structure_file2, output_file)
        elif use_simple:
            calculate_and_align_simple(structure_file1, structure_file2, output_file)
        else:
            calculate_and_align_with_sequence(structure_file1, structure_file2, output_file)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n尝试使用 --simple 参数使用原始方法")
        sys.exit(1)