#!/usr/bin/env python3
"""
Local-Cloud Synchronization Script

Ensures local development environment stays in sync with Streamlit Cloud
and prevents data loss by maintaining proper backups and version control.
"""

import subprocess
import json
import shutil
from datetime import datetime
from pathlib import Path

class LocalCloudSync:
    """Synchronize local development with Streamlit Cloud deployment"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.backup_dir = self.project_root / "data" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self):
        """Create backup of all important data files"""
        print("💾 Creating backup of data files...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)
        
        # Files to backup
        backup_files = [
            "data/enhanced_blind_responses.json",
            "data/ground_truth_answers.json",
            "data/evaluation_questions.yaml",
            "data/shopping_trends.csv",
            "data/Tesla_stock_data.csv"
        ]
        
        backed_up = []
        for file_path in backup_files:
            source = self.project_root / file_path
            if source.exists():
                dest = backup_subdir / source.name
                shutil.copy2(source, dest)
                backed_up.append(file_path)
                print(f"   ✅ {file_path}")
        
        # Create backup manifest
        manifest = {
            "timestamp": timestamp,
            "files_backed_up": backed_up,
            "backup_location": str(backup_subdir)
        }
        
        with open(backup_subdir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Backup created: {backup_subdir}")
        return backup_subdir
    
    def check_git_status(self):
        """Check git status and ensure everything is committed"""
        print("🔍 Checking git status...")
        
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout.strip():
                print("⚠️ Uncommitted changes found:")
                print(result.stdout)
                return False
            else:
                print("✅ Git repository is clean")
                return True
                
        except Exception as e:
            print(f"❌ Error checking git status: {e}")
            return False
    
    def sync_to_cloud(self):
        """Sync local changes to Streamlit Cloud"""
        print("☁️ Syncing to Streamlit Cloud...")
        
        # Check if git is clean
        if not self.check_git_status():
            print("⚠️ Please commit your changes first:")
            print("   git add -A")
            print("   git commit -m 'your commit message'")
            return False
        
        try:
            # Push to origin
            result = subprocess.run(
                ["git", "push", "origin", "master"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                print("✅ Successfully pushed to Streamlit Cloud")
                print("🚀 Streamlit Cloud will auto-deploy in 1-2 minutes")
                return True
            else:
                print(f"❌ Push failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error pushing to cloud: {e}")
            return False
    
    def pull_from_cloud(self):
        """Pull latest changes from Streamlit Cloud"""
        print("⬇️ Pulling latest from Streamlit Cloud...")
        
        try:
            result = subprocess.run(
                ["git", "pull", "origin", "master"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                print("✅ Successfully pulled from Streamlit Cloud")
                print(result.stdout)
                return True
            else:
                print(f"❌ Pull failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error pulling from cloud: {e}")
            return False
    
    def verify_file_integrity(self):
        """Verify all required files are present and valid"""
        print("🔍 Verifying file integrity...")
        
        required_files = {
            "data/enhanced_blind_responses.json": "Enhanced responses",
            "data/ground_truth_answers.json": "Ground truth answers",
            "data/evaluation_questions.yaml": "Evaluation questions",
            "data/shopping_trends.csv": "Retail dataset",
            "data/Tesla_stock_data.csv": "Finance dataset",
            "src/ui/main.py": "Main application"
        }
        
        all_good = True
        for file_path, description in required_files.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                size_kb = full_path.stat().st_size / 1024
                print(f"   ✅ {description}: {size_kb:.1f}KB")
            else:
                print(f"   ❌ {description}: MISSING")
                all_good = False
        
        if all_good:
            print("✅ All files present and accounted for")
        else:
            print("⚠️ Some files are missing - please restore from backup or pull from cloud")
        
        return all_good
    
    def show_sync_status(self):
        """Show current synchronization status"""
        print("\n" + "="*60)
        print("📊 LOCAL-CLOUD SYNC STATUS")
        print("="*60)
        
        # Git status
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            local_commit = result.stdout.strip()[:8]
            
            result = subprocess.run(
                ["git", "rev-parse", "origin/master"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            remote_commit = result.stdout.strip()[:8]
            
            if local_commit == remote_commit:
                print(f"🟢 In Sync: {local_commit}")
            else:
                print(f"🟡 Local:  {local_commit}")
                print(f"🟡 Remote: {remote_commit}")
                print("⚠️ Local and remote are different")
                
        except Exception as e:
            print(f"❌ Cannot determine sync status: {e}")
        
        # File verification
        self.verify_file_integrity()
        
        # Recent backups
        backups = sorted(self.backup_dir.glob("backup_*"), reverse=True)
        if backups:
            latest_backup = backups[0]
            print(f"💾 Latest backup: {latest_backup.name}")
        else:
            print("⚠️ No backups found")
        
        print("="*60)

def main():
    """Main synchronization interface"""
    sync = LocalCloudSync()
    
    print("🔄 Local-Cloud Synchronization Tool")
    print("="*50)
    
    # Show current status
    sync.show_sync_status()
    
    print("\n📋 Available commands:")
    print("1. Create backup")
    print("2. Pull from cloud")
    print("3. Push to cloud")
    print("4. Verify files")
    print("5. Show status")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            sync.create_backup()
        elif choice == "2":
            sync.pull_from_cloud()
        elif choice == "3":
            sync.sync_to_cloud()
        elif choice == "4":
            sync.verify_file_integrity()
        elif choice == "5":
            sync.show_sync_status()
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n🛑 Cancelled")

if __name__ == "__main__":
    main() 