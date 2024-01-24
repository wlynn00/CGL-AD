import os

def rename_file(folder_path):
    print(folder_path)

    for filename in os.listdir(folder_path):
        # num = int((filename.split('.')[0]).split('-')[-1])
        # if 300 <= num <= 399:
        #     continue
        # else:
        #     new_name = filename.replace('attack', 'benign')
        # old_path = os.path.join(folder_path, filename)
        # new_path = os.path.join(folder_path, new_name)
        # os.rename(old_path, new_path)
        if os.path.isfile(os.path.join(folder_path, filename)):
            if 'attack' in filename:
                num = int(filename.split('.')[-1])
                new_name = filename.replace(str(num), str(num+125))
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
            elif 'benign' in filename:
                name = (filename.split('.')[0]).split('-')[-1]
                new_name = filename.replace(name, 'normal')
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)


def decompress_data(scene):
    for i in range(3):
        os.system(
            'tar -zxvf ../../../../dataset/Unicorn/{}/attack/camflow-attack-'.format(scene) + str(i) + '.gz.tar')
    for i in range(13):
        os.system(
            'tar -zxvf ../../../../dataset/Unicorn/{}/benign/camflow-benign-'.format(scene) + str(i) + '.gz.tar')
    os.system('rm error.log')
    os.system('rm parse-error-camflow-*')
    rename_file('./')  # 将攻击图文件后缀改为125-149