import os

scene = 'shellshock'
# scene = 'camflow_apt'
# scene = 'streamspot'

data_root_path = '../data/{}'.format(scene)
base_path = os.path.join(data_root_path, 'base_graphs')
stream_path = os.path.join(data_root_path, 'stream_edges')

os.system('cd {} && mkdir -p graph_sketch\n'.format(data_root_path))
sketch_save_path = os.path.join(data_root_path, 'graph_sketch')


def sort_key(filename):
    parts = filename.split('-')
    number = int(parts[-1].split('.')[0]) if '.' in parts[-1] else int(parts[-1])
    return number


def sorted_listdir(path):
    files = os.listdir(path)
    sorted_files = sorted(files, key=sort_key)
    return sorted_files

fw = open('./analyse_{}.sh'.format(scene), 'w')
fw.write('echo $(date +%F%n%T)\n')
for filename in sorted_listdir(base_path):
    if os.path.isfile(os.path.join(base_path, filename)) and filename.endswith('.txt'):
        num = int((filename.split('.')[0]).split('-')[-1])
        # if num > 299 and num < 400:   # streamspot
        #     name = 'attack'
        if num > 124:   # camflow_apt and shellshock
            name = 'attack'
        else:
            name = 'normal'
        fw.write('bin/unicorn/main filetype edgelist base {}/{} stream {}/stream-{}-{}.txt decay 500 lambda 0.02 '
                 'window 3000 sketch {}/sketch-{}-{}.txt chunkify 1 chunk_size 100\n'.format(base_path, filename,
                                                                                            stream_path, name, num,
                                                                                            sketch_save_path, name,
                                                                                            num))
        # remove tmp files
        fw.write('rm -rf {}/base-{}-{}.txt.*\n'.format(base_path, name, num))
        fw.write('rm -rf {}/base-{}-{}.txt_*\n'.format(base_path, name, num))
fw.write('echo $(date +%F%n%T)\n')
fw.close()