#!/usr/bin/env python
docstring='''DomainParser.py 16pkA.pdb > domain_list.txt
    run DomainParser on target PDB 16pkA.pdb to paritition it into domains. 
    Domains will be listed in "domain_list.txt" in the following format:
        basename	length	domain_number	list_of_domain
    (where "basename" is the basename of input file name (without extension).
    "list_of_domain" is space separated list of PDB residue number
    ranges quoted inside parathesis)
    or, in case no domain are found:
        basename	length	1

    MODIFICATION: Domains are now sorted by their starting residue position in the chain,
    ensuring that domain file names (_d_1, _d_2, etc.) reflect the actual order
    of domains along the protein sequence. Additionally, fragments within each domain
    are sorted to maintain correct residue ordering.

options:
    -execpath=./domainparser2.LINUX
        path to DomainParser executable. By default it is guessed by 
        location of this script
    -dssp_path=./dssp
        path to DSSP executable. By default it is guessed by
        location of this script
    -pulchra_path=./pulchra
        path to pulchra executable. By default it is guessed by 
        location of this script. pulchra is used to construct full atom
        model from backbone model when the input structure contains too
        few atoms.
    -log=DomainParser.log
        path to DomainParser output log. "-" for stdout
    -debug
        enable debug mode with verbose output
'''
import sys,os
import shutil
import random
import subprocess
import Bio.PDB
import re
import warnings

segment_pattern=re.compile("([-]{0,1}\d+)[A-Za-z]{0,1}[-]([-]{0,1}\d+)[A-Za-z]{0,1}")

def check_executable(exec_path, name):
    '''检查可执行文件是否存在和可执行'''
    if not os.path.isfile(exec_path):
        print(f"ERROR: {name} executable not found at: {exec_path}")
        return False
    if not os.access(exec_path, os.X_OK):
        print(f"ERROR: {name} executable not executable: {exec_path}")
        return False
    print(f"Found {name} executable: {exec_path}")
    return True

def validate_pdb_basic(pdb_file):
    '''基本PDB文件验证'''
    try:
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        
        atom_count = sum(1 for line in lines if line.startswith('ATOM'))
        print(f"PDB file validation: {len(lines)} lines, {atom_count} ATOM records")
        
        if atom_count == 0:
            print("ERROR: No ATOM records found in PDB file")
            return False
        return True
    except Exception as e:
        print(f"ERROR reading PDB file: {e}")
        return False

def parse_domain_residues(domain_string, debug=False):
    '''解析结构域字符串，返回残基列表和起始残基号
    改进版本：确保片段按照序列顺序排列'''
    segments = []
    try:
        # 解析所有片段
        for segment in domain_string.strip('()').split(';'):
            if not segment.strip():  # 跳过空片段
                continue
            matches = segment_pattern.findall(segment)
            if matches:
                start, end = int(matches[0][0]), int(matches[0][1])
                segments.append((start, end))
                if debug:
                    print(f"    Parsed fragment: {start}-{end}")
        
        if not segments:
            if debug:
                print(f"    Warning: No valid segments found in {domain_string}")
            return [], None
        
        # 按起始位置排序片段（关键改进）
        segments.sort(key=lambda x: x[0])
        if debug:
            print(f"    Sorted fragments: {segments}")
        
        # 按排序后的顺序合并残基
        resi_list = []
        for start, end in segments:
            resi_list.extend(range(start, end + 1))
        
        if debug:
            print(f"    Final residue list: {len(resi_list)} residues from {min(resi_list)} to {max(resi_list)}")
        
        return resi_list, min(resi_list)
        
    except Exception as e:
        print(f"Warning: Error parsing domain {domain_string}: {e}")
        return [], None

def DomainParser(pdb_file,execpath="domainparser2.LINUX",
    dssp_path="dssp", pulchra_path="pulchra", debug=False):
    '''run DomainParser executable "execpath" using DSSP executable
    "dssp_path" on pdb file "pdb_file"
    '''
    if debug:
        print(f"=== DEBUG: Starting DomainParser for {pdb_file} ===")
    
    # 验证输入文件
    if not validate_pdb_basic(pdb_file):
        return ''
    
    # 检查可执行文件
    if not check_executable(execpath, "DomainParser"):
        return ''
    if not check_executable(dssp_path, "DSSP"):
        return ''
    # pulchra是可选的，只在需要时检查
    
    #### make temporary folder ####
    basename=os.path.basename(pdb_file)
    tmp_dir="/tmp/"+os.getenv("USER","user")+"/DomainParser"+ \
        str(random.randint(1000,9999))+basename.split('.')[0]+'/'
    
    if debug:
        print(f"DEBUG: Creating temporary directory: {tmp_dir}")
    
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    tmp_pdb=tmp_dir+'xxxx.pdb'

    #### parse PDB files ####
    if debug:
        print(f"DEBUG: Parsing PDB file with BioPython...")
    
    try:
        # 抑制BioPython警告
        warnings.filterwarnings("ignore", module="Bio.PDB.PDBParser")
        warnings.filterwarnings("ignore", module="Bio.PDB.Atom")
        warnings.filterwarnings("ignore", module="Bio.PDB.PDBIO")
        
        struct = Bio.PDB.PDBParser(PERMISSIVE=1).get_structure(pdb_file,pdb_file)
        model=struct[0]
        chain=[c for c in model][0]
        chain.id=chain.id[0].upper()
        io=Bio.PDB.PDBIO()
        io.set_structure(chain)
        io.save(tmp_pdb)
        chain_id=chain.id
        
        warnings.resetwarnings()
        
        if debug:
            print(f"DEBUG: BioPython parsing successful, chain ID: {chain_id}")
            print(f"DEBUG: Temporary PDB saved: {tmp_pdb}")
            
    except Exception as e:
        print(f"ERROR: BioPython parsing failed: {e}")
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        return ''

    #### run DomainParser ####
    cmd=' '.join(['cd',tmp_dir,';',
        'export DSSP_PATH='+dssp_path,';',
        execpath,'xxxx'+chain_id
        ])
    
    if debug:
        print(f"DEBUG: Running command: {cmd}")
    
    try:
        p=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,universal_newlines=True)
        stdout,stderr=p.communicate()
        
        if debug:
            print(f"DEBUG: Command return code: {p.returncode}")
            print(f"DEBUG: stdout length: {len(stdout) if stdout else 0}")
            print(f"DEBUG: stderr length: {len(stderr) if stderr else 0}")
            if stderr:
                print(f"DEBUG: stderr content: {stderr[:200]}...")
                
    except Exception as e:
        print(f"ERROR: Failed to run DomainParser command: {e}")
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        return ''

    #### parse output ####
    if not stdout.strip():
        if debug:
            print("DEBUG: No stdout from DomainParser, checking stderr...")
        
        if stderr.startswith("Missing sidechain atoms for"):
            if debug:
                print("DEBUG: Missing sidechain atoms, trying pulchra reconstruction...")
                
            # 检查pulchra是否可用
            if not check_executable(pulchra_path, "Pulchra"):
                print("ERROR: Pulchra needed but not available")
                shutil.rmtree(tmp_dir)
                return ''
            
            fp=open(tmp_pdb,'r')
            txt=''.join([line+'\n' for line in fp.read().splitlines() if \
                line.startswith('ATOM  ') and line[12:16]==' CA '])
            fp.close()
            fp=open(tmp_pdb,'w')
            fp.write(txt)
            fp.close()

            pul_cmd=' '.join(['cd',tmp_dir,';',pulchra_path,'-epc xxxx.pdb'])
            if debug:
                print(f"DEBUG: Running pulchra: {pul_cmd}")
                
            try:
                subprocess.Popen(pul_cmd, stdout=subprocess.PIPE, shell=True,
                    universal_newlines=True).communicate()
            except Exception as e:
                print(f"ERROR: Pulchra failed: {e}")
                shutil.rmtree(tmp_dir)
                return ''
                
            # 查找重建的文件
            rebuilt_files = ["xxxx.rebuilt.pdb", "rebuilt_xxxx.pdb", "pul_xxxx.pdb"]
            pdb_file_rebuilt = None
            for rebuilt_file in rebuilt_files:
                if os.path.isfile(tmp_dir + rebuilt_file):
                    pdb_file_rebuilt = tmp_dir + rebuilt_file
                    if debug:
                        print(f"DEBUG: Found rebuilt file: {pdb_file_rebuilt}")
                    break
            
            if not pdb_file_rebuilt:
                print("ERROR: Pulchra reconstruction failed, no rebuilt file found")
                if debug:
                    files = os.listdir(tmp_dir)
                    print(f"DEBUG: Files in tmp_dir: {files}")
                shutil.rmtree(tmp_dir)
                return ''
            
            # 重新处理重建的文件
            try:
                warnings.filterwarnings("ignore",module="Bio.PDB.PDBIO")
                warnings.filterwarnings("ignore",module="Bio.PDB.PDBParser")
                warnings.filterwarnings("ignore",module="Bio.PDB.Atom")
                struct = Bio.PDB.PDBParser(PERMISSIVE=1
                    ).get_structure(pdb_file_rebuilt,pdb_file_rebuilt)
                model=struct[0]
                chain=[c for c in model][0]
                chain.id=chain_id
                io=Bio.PDB.PDBIO()
                io.set_structure(chain)
                io.save(tmp_pdb)
                warnings.resetwarnings()
            except Exception as e:
                print(f"ERROR: Failed to process rebuilt PDB: {e}")
                shutil.rmtree(tmp_dir)
                return ''

            # 重新运行DomainParser
            if debug:
                print("DEBUG: Re-running DomainParser with rebuilt structure...")
                
            try:
                p=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,universal_newlines=True)
                stdout,stderr=p.communicate()
                
                if debug:
                    print(f"DEBUG: Second run return code: {p.returncode}")
                    
            except Exception as e:
                print(f"ERROR: Second DomainParser run failed: {e}")
                shutil.rmtree(tmp_dir)
                return ''

            if not stdout.strip():
                print(f"ERROR: DomainParser failed even after reconstruction. stderr: {stderr}")
                shutil.rmtree(tmp_dir)
                return ''
        else:
            print(f"ERROR: DomainParser failed. stderr: {stderr}")
            shutil.rmtree(tmp_dir)
            return ''
    elif stdout.startswith("Something is wrong"):
        print(f"ERROR! DomainParser reports: Something is wrong with {pdb_file}")
        shutil.rmtree(tmp_dir)
        return ''
        
    if debug:
        print(f"DEBUG: DomainParser output: {stdout[:100]}...")
        
    output_list=stdout.split()
    if len(output_list) < 3:
        print("ERROR! Invalid DomainParser output format")
        print(f"Output was: {stdout}")
        shutil.rmtree(tmp_dir)
        return ''
        
    target,seqlen,domain_num=output_list[:3]
    domain_list=output_list[3:]
    
    if debug:
        print(f"DEBUG: Found {domain_num} domains: {domain_list}")

    #### parse domains and sort by residue position ####
    domain_info_list = []
    for domain_idx, domain in enumerate(domain_list):
        if debug:
            print(f"  Processing domain {domain_idx+1}: {domain}")
        
        resi_list, min_resi = parse_domain_residues(domain, debug)
        
        if resi_list and min_resi is not None:
            domain_info_list.append({
                'domain_string': domain,
                'resi_list': resi_list,
                'min_resi': min_resi,
                'max_resi': max(resi_list),
                'original_idx': domain_idx,
                'fragment_count': domain.count('-')
            })
            if debug:
                fragment_count = domain.count('-')
                print(f"    Domain {domain_idx+1}: {fragment_count} fragments, "
                      f"spans residues {min_resi}-{max(resi_list)} (original order)")

    # 按照最小残基号排序
    domain_info_list.sort(key=lambda x: x['min_resi'])
    
    if debug:
        print("DEBUG: Domains sorted by residue position:")
        for i, domain_info in enumerate(domain_info_list):
            print(f"  New order {i+1}: residues {domain_info['min_resi']}-{domain_info['max_resi']} "
                  f"({domain_info['fragment_count']} fragments, "
                  f"was domain {domain_info['original_idx']+1})")

    #### output individual domain in correct order ####
    sorted_domain_strings = []  # 保存排序后的结构域字符串用于最终输出
    
    for new_idx, domain_info in enumerate(domain_info_list):
        domain_file = basename.split('.')[0] + "_d_" + str(new_idx + 1) + ".pdb"
        resi_list = domain_info['resi_list']
        sorted_domain_strings.append(domain_info['domain_string'])
        
        if resi_list:
            class ResiSelect(Bio.PDB.Select): # class to select domain
                def accept_residue(self,residue):
                    return 1 if residue.id[1] in resi_list else 0

            try:
                io.save(domain_file,ResiSelect())
                sys.stdout.write(domain_file+'\n')
                if debug:
                    print(f"DEBUG: Saved domain file: {domain_file} "
                          f"(residues {domain_info['min_resi']}-{domain_info['max_resi']}, "
                          f"{domain_info['fragment_count']} fragments)")
            except Exception as e:
                print(f"Warning: Error saving domain file {domain_file}: {e}")

    #### cleanup temporary folder ####
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
        
    if debug:
        print("=== DEBUG: DomainParser completed successfully ===")
        print("=== DEBUG: Domain ordering summary ===")
        print("Original DomainParser order -> New sequence-based order:")
        for i, domain_info in enumerate(domain_info_list):
            print(f"  Domain {domain_info['original_idx']+1} -> Domain {i+1} "
                  f"(residues {domain_info['min_resi']}-{domain_info['max_resi']}, "
                  f"{domain_info['fragment_count']} fragments)")
        print("=== Fragment-level details ===")
        for i, domain_info in enumerate(domain_info_list):
            print(f"  Domain {i+1}: {domain_info['domain_string']}")
            resi_list, _ = parse_domain_residues(domain_info['domain_string'], debug=False)
            if len(resi_list) > 10:
                print(f"    Residues: {resi_list[:5]}...{resi_list[-5:]} ({len(resi_list)} total)")
            else:
                print(f"    Residues: {resi_list}")
        
    # 返回结果，使用排序后的结构域列表
    return '\t'.join([
        basename.split('.')[0], seqlen, domain_num, ' '.join(sorted_domain_strings)])

def locate_DomainParser():
    '''locate the location of DomainParser exectuable'''
    possible_names = ["domainparser2", "domainparser2.LINUX"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 首先在脚本同目录查找
    for name in possible_names:
        execpath = os.path.join(script_dir, name)
        if os.path.isfile(execpath):
            return execpath
    
    # 然后在当前目录查找
    for name in possible_names:
        if os.path.isfile(name):
            return os.path.abspath(name)
    
    # 默认返回
    return "domainparser2"

if __name__=="__main__":
    execpath=locate_DomainParser()
    dssp_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
        "dssp")
    pulchra_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
        "pulchra")
    log='DomainParser.log'
    debug = False

    if len(sys.argv)<2:
        sys.stderr.write(docstring)
        sys.exit(1)

    argv=[]
    for arg in sys.argv[1:]:
        if arg.startswith("-execpath="):
            execpath=os.path.abspath(arg[len("-execpath="):])
        elif arg.startswith("-dssp_path="):
            dssp_path=os.path.abspath(arg[len("-dssp_path="):])
        elif arg.startswith("-pulchra_path="):
            pulchra_path=os.path.abspath(arg[len("-pulchra_path="):])
        elif arg.startswith("-log="):
            log=arg[len("-log="):]
        elif arg == "-debug":
            debug = True
        elif arg.startswith("-"):
            sys.stderr.write("ERROR! Unknown option %s\n"%arg)
            sys.exit(1)
        else:
            if not os.path.isfile(arg):
                sys.stderr.write("ERROR! No such file %s\n"%arg)
                sys.exit(1)
            argv.append(arg)
    
    if debug:
        print("=== DEBUG MODE ENABLED ===")
        print(f"DomainParser executable: {execpath}")
        print(f"DSSP path: {dssp_path}")  
        print(f"Pulchra path: {pulchra_path}")
        print(f"Log file: {log}")
        print(f"Input files: {argv}")
    
    txt=''
    for arg in argv:
        if debug:
            print(f"\n=== Processing {arg} ===")
        result = DomainParser(arg,execpath,dssp_path,pulchra_path,debug)
        if result:
            txt+=result+'\n'
        elif debug:
            print(f"WARNING: No result returned for {arg}")

    if log and log!='-':
        try:
            with open(log,'w') as fp:
                fp.write(txt)
            print(f"Results written to log file: {log}")
            if debug and txt:
                print(f"Log content preview: {txt[:100]}...")
        except Exception as e:
            print(f"Failed to write log file: {e}")
            sys.stdout.write(txt)
    else:
        sys.stdout.write(txt)