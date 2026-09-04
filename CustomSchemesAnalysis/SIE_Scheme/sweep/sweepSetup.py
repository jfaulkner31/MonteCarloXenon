import CustomSchemes
from pathlib import Path
import shutil

# Removing folders and their contents
def remove_folders():
  numIter = [100,100,18,12,7]
  m = [1,2,3,4,5,6]
  f = [1,1,1.5,2.0,4.0]
  n = [100000, 10000, 10000, 10000, 10000] # npg
  sims = 20 

  for idx, _ in enumerate(numIter):
    the_num_iter = numIter[idx]
    the_f = f[idx]
    the_npg = n[idx]
    for the_m in m:
      for the_sim_num in range(sims):
        folder_name = f"aa{the_m}_f{the_f}_s{the_sim_num+1}_npg{the_npg}"
        shutil.rmtree(folder_name)

def copy_replace_lines(filename, lines_to_search, lines_to_write, the_filename):
  src = Path(filename)
  dst = Path(the_filename)

  if len(lines_to_search) != len(lines_to_write):
      raise ValueError("lines_to_search and lines_to_write must have the same length")

  replacements = {
      s.strip(): w
      for s, w in zip(lines_to_search, lines_to_write)
  }

  shutil.copy2(src, dst)

  lines = dst.read_text().splitlines()

  new_lines = [
      replacements.get(line.strip(), line)
      for line in lines
  ]

  dst.write_text("\n".join(new_lines) + "\n")

"""
Actually running stuff now.
"""

numIter = [100,100,18,12,7]
m = [1,2,3,4,5,6]
f = [1,1,1.5,2.0,4.0]
n = [100000, 10000, 10000, 10000, 10000] # npg
sims = 20 

for idx, _ in enumerate(numIter):
  the_num_iter = numIter[idx]
  the_f = f[idx]
  the_npg = n[idx]
  for the_m in m:
    for the_sim_num in range(sims):
      folder_name = f"aa{the_m}_f{the_f}_s{the_sim_num+1}_npg{the_npg}"

      # Make the directory
      Path(folder_name).mkdir(parents=True, exist_ok=True)

      # The filename
      src = str(Path(CustomSchemes.__file__).parent / "aa_singlestep_SWEEP.py")

      # Destination
      dst = Path(folder_name) / "aa_singlestep_SWEEP.py"

      # Copy and replace lines : numIter, m, f, n
      lines_to_search = ["THE_NUMBER_OF_SOLVES = None", "THE_ANDERSON_ORDER = None", "THE_SCALE_NPG = None", "THE_STARTING_NPG = None"]
      lines_to_write =  [f"THE_NUMBER_OF_SOLVES = {int(the_num_iter)}", f"THE_ANDERSON_ORDER = {the_m}", f"THE_SCALE_NPG = {the_f}", f"THE_STARTING_NPG = {the_npg}"]
      copy_replace_lines(filename=src, lines_to_search=lines_to_search, lines_to_write=lines_to_write, the_filename=dst)

      # Copy the slurm script now
      # shutil.copy(src="job.slurm", dst=Path(folder_name) / "job.slurm")
      copy_replace_lines(filename="job.slurm", lines_to_search=["#SBATCH --job-name=JOB_NAME"],
                         lines_to_write=[f"#SBATCH --job-name={folder_name}"], the_filename=Path(folder_name) / "job.slurm")
