nohup python main.py --config config/config.yaml --mode train > training_log.txt 2>&1 &

# View the entire log file
  cat training_log.txt

  # View the last 20 lines
  tail -20 training_log.txt

  # View the log in real-time as it updates (most useful)
  tail -f training_log.txt

  # Press Ctrl+C to exit the tail -f view

  The tail -f command is particularly useful as it will continuously show new lines as they're added to the log file,
  allowing you to monitor training progress in real-time.

  You can also check if the process is still running using:

  # Find your Python process
  ps aux | grep "[p]ython main.py"

  # Or get the process ID from the nohup.out file that was created
  cat nohup.out

  # To kill the background nohup process:

  1. Find the process ID (PID):
  ps aux | grep "[p]ython main.py"
  1. This shows something like:
  username  12345  2.0  1.5  123456 78910 ?   S    06:38   0:12 python main.py --config config/config.yaml --mode train
  1. The number after your username (12345 in this example) is the PID.
  2. Kill the process:
  kill 12345  # Replace with your actual PID
  3. If it doesn't stop, force kill it:
  kill -9 12345  # Replace with your actual PID
  4. Alternatively, kill all Python processes (use with caution if you have other Python scripts running):
  pkill python

  The process will be terminated, and you can confirm it's gone by running the grep command again.

  # Run with lightweight config
  nohup python main.py --config config/light_config.yaml --mode train > light_training_log.txt 2>&1 &
