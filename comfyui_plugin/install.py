#!/usr/bin/env python3
import os
import sys
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description="Install AniGen ComfyUI plugin installer")
    parser.add_argument("--comfyui-dir", required=True, help="Path to ComfyUI directory")
    parser.add_argument("--link", action="store_true", help="Use symbolic link instead of copying")
    args = parser.parse_args()
    
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.join(args.comfyui_dir, "custom_nodes")
    target_dir = os.path.join(custom_nodes_dir, "AniGen")
    
    if not os.path.exists(custom_nodes_dir):
        print(f"Error: ComfyUI custom_nodes directory not found: {custom_nodes_dir}")
        return 1
    
    # Remove existing installation if present
    if os.path.exists(target_dir):
        print(f"Removing existing installation at {target_dir}")
        if os.path.islink(target_dir):
            os.unlink(target_dir)
        else:
            shutil.rmtree(target_dir)
    
    # Install
    if args.link:
        print(f"Creating symbolic link from {plugin_dir} to {target_dir}")
        os.symlink(plugin_dir, target_dir)
    else:
        print(f"Copying {plugin_dir} to {target_dir}")
        shutil.copytree(plugin_dir, target_dir)
    
    print("Installation complete!")
    print("Please restart ComfyUI to load the plugin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
