# coding:utf-8
import os
import re
import tensorflow as tf

base_dir = '/tmp/imagenet'

def main(argv=None):
    node_lookup = node_dict()
    #モデル定義ファイルからグラフを復元
    with tf.gfile.FastGFile(os.path.join(base_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    #同梱されている画像の読み込みとモデルへのデータ入力
    image_data = tf.gfile.FastGFile(os.path.join(base_dir, 'cropped_panda.jpg'), 'rb').read()
    with tf.Session() as sess:
        #最終出力のTensorを取得
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        #画像を入力して，計算結果を取得
        predictions = sess.run(tf.squeeze(softmax_tensor), feed_dict={
            'DecodeJpeg/contents:0' : image_data
        })
        top_k = predictions.argsort()[-3:][::-1]
        print(predictions.argsort()[-3:])
        print(node_lookup[169])
        for node_id in top_k:
            human_string = node_lookup[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

#ラベル定義ファイルから認識結果のラベル番号とラベル名の対応辞書を生成
def node_dict():
    label_lookup_path = os.path.join(base_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    uid_lookup_path = os.path.join(base_dir, 'imagenet_synset_to_human_label_map.txt')

    uid_to_human = {}
    p = re.compile(r'[n\d]*[\S,]*')
    for line in tf.gfile.GFile(uid_lookup_path).readlines():
        parsed_items = p.findall(line)
        uid = parsed_items[0]
        human_string = parsed_items[2]
        uid_to_human[uid] = human_string
    node_id_to_uid = {}
    for line in tf.gfile.GFile(label_lookup_path).readlines():
        if line.startswith('  target_class:'):
            target_class = int(line.split(': ')[1])
        if line.startswith('  target_class_string:'):
            target_class_string = line.split(': ')[1]
            node_id_to_uid[target_class] = target_class_string[1:-2]
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
        name = uid_to_human[val]
        node_id_to_name[key] = name
    return node_id_to_name

if __name__ == '__main__':
    tf.app.run()


### わからない関数 ###
# tf.gfile.FastGFile：File IO wrapper without thread locking URL: https://www.tensorflow.org/api_docs/python/tf/gfile/FastGFile
# fileと同じ容量で使える．
# ParseFromString： URL: https://www.tensorflow.org/extend/tool_developers/
# import_graph_def： URL1: https://www.tensorflow.org/api_docs/python/tf/import_graph_def
# sess.graph.get_tensor_by_name：
# tf.squeeze：大きさが1の次元を削除する．引数をリストで与えるとそのリストの要素のインデックスの要素の大きさが１かどうかを判定
# argsort：pythonの関数，ソートされたインデックスを返す．
# startswith：pythonの関数，指定の文字列を含むかどうかの真偽を返す．
#
