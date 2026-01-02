import argparse
import sys
import os
from dotenv import load_dotenv
from src.utils.logger import log_experiment
from src.services.file_handler import file_service

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        print(f"❌ Dossier {args.target_dir} introuvable.")
        sys.exit(1)

    print(f"🚀 DEMARRAGE SUR : {args.target_dir}")
    log_experiment("System", "STARTUP", f"Target: {args.target_dir}", "INFO")
    print("✅ MISSION_COMPLETE")

if __name__ == "__main__":
    main()        #comment this line if you wanna test the file service
    
    # ptf = './sandbox/idk.py'
    # txt = file_service.read_file_to_text(ptf)
    # print(txt)
    # txt += '\nprint("ooooh I am working")'
    # file_service.write_text_to_file(ptf, txt)

    # print('done')
    pass
    