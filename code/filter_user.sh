
percent=5
python filter_user.py --input_file ../dataset/Pixel200K_ori.csv --input_file2 ../information/Pixel200K_ori.csv --output_file ../dataset/Pixel200K_${percent}_percent.csv --output_file2 ../information/Pixel200K_${percent}_percent.csv --percent ${percent}
# python filter_user.py --input_file ../dataset/microlens.csv --input_file2 ../information/microlens.csv --output_file ../dataset/microlens_${percent}_percent.csv --output_file2 ../information/microlens_${percent}_percent.csv --percent ${percent}
