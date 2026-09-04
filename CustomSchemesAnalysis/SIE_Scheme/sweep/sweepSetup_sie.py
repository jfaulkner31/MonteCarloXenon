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


def remove_sie_folders():
  sims = 20
  numIter = [1000, 100, 10]
  npg = [10000, 100000, 1000000]
  for idx, _ in enumerate(numIter):
    the_num_iter = numIter[idx]
    the_npg = npg[idx]
    for s in range(sims):
      folder_name = f"sie_s{s+1}_npg{the_npg}_i{the_num_iter}"
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

sims = 20
numIter = [1000, 100, 10]
npg = [10000, 100000, 1000000]

for idx, _ in enumerate(numIter):
  the_num_iter = numIter[idx]
  the_npg = npg[idx]
  for s in range(sims):
    folder_name = f"sie_s{s+1}_npg{the_npg}_i{the_num_iter}"
    # Make the directory
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    src = str(Path(CustomSchemes.__file__).parent / "sie_singlestep_SWEEP.py")

    dst = Path(folder_name) / "sie_singlestep_SWEEP.py"

    lines_to_search = ["STARTING_PARTICLES = None", "NSOLVES = None"]
    lines_to_write =  [f"STARTING_PARTICLES = {int(the_npg)}", f"NSOLVES = {the_num_iter}"]
    copy_replace_lines(filename=src, lines_to_search=lines_to_search, lines_to_write=lines_to_write, the_filename=dst)

    copy_replace_lines(filename="sie_job.slurm", lines_to_search=["#SBATCH --job-name=JOB_NAME"],
                         lines_to_write=[f"#SBATCH --job-name={folder_name}"], the_filename=Path(folder_name) / "sie_job.slurm")
