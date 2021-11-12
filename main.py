from tkinter import filedialog
import codecs
import time
from queue import Queue
import numpy as np
from numba import jit
from numba.typed import List


# this is the merge sort function which sorts the given node array bases on each nodes hash value
# it does this by deviding the array recursivly untill each subarray is sorted then merges each
# subarray to make bigger arrays untill we have the original array
def merge_sort(my_list):
    if len(my_list) > 1:
        mid = len(my_list) // 2
        left = my_list[:mid]
        right = my_list[mid:]


        merge_sort(left)
        merge_sort(right)

        i = 0
        j = 0

        k = 0

        while i < len(left) and j < len(right):
            if left[i].__hash__() <= right[j].__hash__():
                my_list[k] = left[i]
                i += 1
            else:
                my_list[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            my_list[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            my_list[k] = right[j]
            j += 1
            k += 1



class GetDataAndBuildTree:
    file = None
    cost = [[]]
    roots = [[]]
    nodes = []
    freq = []
    root = None
    tree_file = None

# this method is for graphical input taking from the user for dictionary
    @classmethod
    def get_txt_location(cls):
        return filedialog.askopenfilename(title='select your dictionary', filetypes=(('txt files', '*.txt'), ('all files', '*.*')))

# this method is responsible for reading dictionary data into an array of node only one thing is diffrent which is that
# frequancies are stored in a comulative way meaning each frequency is the sum of previous one plus itself
# this way of storing reduses the need for calculating the sum each time in the obst algorithm which results
# in a o of 4 time complexity which is terrible
    @classmethod
    def read_txt(cls):
        cls.file = codecs.open(cls.get_txt_location(), 'r', encoding='utf-8-sig')
        cls.nodes.append(Node("", "", float(0)))
        while True:
            line_pointer = cls.file.readline().split(" ")
            if line_pointer == ['']:
                break
            word = line_pointer[0]
            meaning = line_pointer[1]
            if len(cls.nodes) != 1:
                frequency = float(line_pointer[2].replace("\r\n", "")) + cls.nodes[-1].freq
            else:
                frequency = float(line_pointer[2].replace("\r\n", ""))

            cls.nodes.append(Node(word, meaning, frequency))
        cls.file.close()
        dimention = len(cls.nodes)
        # cls.cost = [[float(0) for i in range(len(cls.nodes))] for j in range(len(cls.nodes))]
        # cls.roots = [[int(0) for i in range(len(cls.nodes))] for j in range(len(cls.nodes))]
        # cls.freq = [float(0) for i in range(dimention)]
        cls.cost = np.zeros((dimention, dimention))
        cls.roots = np.zeros((dimention, dimention), dtype=int)
        cls.freq = np.zeros(dimention)
        cls.freq[0] = 0

# in this main method first we try to open tree file so if the program was run before we have a tree file
# and we can just restore the tree without extra cost in the else block otherwise if the file doesn't exitst
# first we graphicly take input dictionary from user then read them and store these data in node objects
# after we're done with that we should sort the nodes array because the obst algorithm needs a sorted array
# to oporate after that we call find_tree function which finds the obst and then we build and store the built
# tree

    @classmethod
    def main(cls):
        try:
            cls.file = codecs.open("tree.txt", 'r', encoding='utf-8-sig')
        except FileNotFoundError:
            cls.read_txt()
            merge_sort(cls.nodes)
            cls.comulitive_freq()
            cls.find_tree(len(cls.nodes), cls.cost, cls.roots, cls.freq)
            cls.root = cls.nodes[int(cls.roots[0][len(cls.nodes) - 1])]
            cls.build_tree(0, len(cls.nodes) - 1, cls.root, cls.roots[0][len(cls.nodes) - 1])
            cls.store_tree()
            return cls.root
        else:
            return cls.restore_tree()






    @classmethod
    def comulitive_freq(cls):
        for i in range(len(cls.nodes) - 1):
            cls.nodes[i+1].freq = cls.nodes[i].freq + cls.nodes[i+1].freq
            cls.freq[i + 1] = cls.nodes[i + 1].freq



    @classmethod
    def weight_sum(cls, i, j):
        return cls.nodes[j].freq - cls.nodes[i].freq

    # this is the function responsible for finding obst it find's it by finding most optimal solution for each sub
    # problem then in each step it calculates the cost and moves diagonaly through the matrix to find all sub problems
    # if you notice the code here is designed to not use object nodes and also uses static c-like arrays this is for the
    # performance reasons i'll explain in depth about performance and what i did to run the code fast in the video
    @staticmethod
    @jit(nopython=True)
    def find_tree(dimention, cost, roots, freq):
        for i in range(0, dimention):
            cost[i][i] = 0
            roots[i][i] = -1

        for j in range(1, dimention):
            for i in range(0, dimention - j):
                temp_min = 9223372036854775807
                for k in range(i+1, i+j+1):
                    new_cost = cost[i][k-1] + cost[k][i + j]
                    if temp_min > new_cost:
                        temp_min = new_cost
                        roots[i][i + j] = k
                cost[i][i + j] = temp_min + freq[i + j] - freq[i]
    # def find_tree(dimention, cost, roots):
    #     for i in range(0, len(cls.nodes)):
    #         cls.cost[i][i] = 0
    #         cls.roots[i][i] = -1
    #
    #     for j in range(1, len(cls.nodes)):
    #         for i in range(0, len(cls.nodes) - j):
    #             temp_min = 9223372036854775807
    #             for k in range(i+1, i+j+1):
    #                 new_cost = cls.cost[i][k-1] + cls.cost[k][i + j]
    #                 if temp_min > new_cost:
    #                     temp_min = new_cost
    #                     cls.roots[i][i + j] = k
    #             cls.cost[i][i + j] = temp_min + cls.nodes[i + j].freq - cls.nodes[i].freq



    # this function using root matrix builds the tree the method is that first we checked for the root then we're left
    # with two subarrays each of these subarrays have their own optimal root and each have their own subarray based of
    # their root which is found in root matrix then we recursivly divide the nodes array based of the index of optimal
    # bst in the root matrix and store the answer to them in left an right sub tree


    @classmethod
    def build_tree(cls, i, j, node, index):
        if i == j:
            node.left_child = Node("", "?", "0", 0)
            node.right_child = Node("", "?", "0", 0)
            return
        if i == index - 1:
            node.left_child = Node("", "?", "0", 0)
        else:
            node.left_child = cls.nodes[cls.roots[i][index - 1]]

        if index == j:
            node.right_child = Node("", "?", "-1", 0)
        else:
            node.right_child = cls.nodes[cls.roots[index][j]]


        cls.build_tree(i, index - 1, node.left_child, cls.roots[i][index - 1])
        cls.build_tree(index, j, node.right_child, cls.roots[index][j])


    # for storing the tree i used bfs traversal and store them in a file later on we can use 2*index +1 and +2 to access
    # left and right child of each node to construct the tree back into memory also the tree is designed to be complete
    # by creating empty nodes for each part that lacks a node in this way once we translate and if we hit one of these
    # nodes we know the word doesn't exist in the tree


    @classmethod
    def store_tree(cls):
        queue = Queue()
        tree = codecs.open("tree.txt", 'w', encoding='utf-8-sig')
        node = cls.root
        queue.put(node)
        while queue.empty() is False:
            node = queue.get_nowait()
            if node is None:
                break
            if node.obj_hash != 0:
                tree.write(node.value + " " + node.translation + " " + str(node.obj_hash) + " " + '\n')
            else:
                tree.write("x x x" + "\n")
            queue.put(node.left_child)
            queue.put(node.right_child)

        tree.close()


    # here as mentioned above we use the given formula to traverse file in bfs manner and putting children in a queue to
    # do the same for each child if we encounter x x x in the file it means we have an empty leaf node we also don't
    # push empty nodes into the queue for that they are done and we don't need children for them


    @classmethod
    def restore_tree(cls):
        store_index = 0
        while True:
            node = cls.file.readline().split(" ")
            if node == ['']:
                break
            if node[0] != 'x':
                cls.nodes.append(Node(node[0], node[1], "", float(node[2]), store_index))
                store_index += 1
            else:
                cls.nodes.append(Node("", "?", "0", float(0), store_index))
                store_index += 1
        cls.file.close()

        queue = Queue()
        node = cls.nodes[0]
        queue.put(node)
        while queue.empty() is False:
            node = queue.get_nowait()
            if 2 * node.index + 1 < len(cls.nodes):
                node.left_child = cls.nodes[2 * node.index + 1]
            if 2 * node.index + 2 < len(cls.nodes):
                node.right_child = cls.nodes[2 * node.index + 2]
            if node.left_child is not None:
                queue.put(node.left_child)
            if node.right_child is not None:
                queue.put(node.right_child)

        cls.root = cls.nodes[0]
        return cls.root


# this class stores all the node data and it's so called a data class this class also uses a hash for each node based on
# english word of each node hashing this way doesn't create coliton because we don't want to store our data in a limited
# array and therfore we don't return remainder of diviton by arr_length these hashes are then used to find_tree in obst
# and make the tree binary

class Node:
    def __init__(self, value, translation, freq, obj_hash=None, index=0):
        self.value = value
        self.translation = translation
        self.freq = freq
        self.obj_hash = obj_hash
        self.index = index
        self.left_child = None
        self.right_child = None


    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        if self.obj_hash is None:
            the_hash = 0
            for i in range(0, len(self.value)):
                the_hash += ord(self.value[i])*pow(10, i)
            self.obj_hash = the_hash
            return the_hash
        else:
            return self.obj_hash


# translate class doesn't do much it's responsible for hashing the word that neads to be translated and checking the
# hashed value in the tree and traversing the tree in search_tree method once found it prints the translation in the
# console and if we hit one of those empty nodes it simply prints a ?
class Translate:

    tree_root = None
    input = None

    @classmethod
    def hash_of_string(cls, word):
        the_hash = 0
        for i in range(0, len(word)):
            the_hash += ord(word[i]) * pow(10, i)
        return the_hash

    @classmethod
    def search_tree(cls, word_hash, node):
        if node is None:
            print("?", end=' ')
        elif node.obj_hash == word_hash and node.obj_hash != 0:
            print(node.translation, end=' ')
        elif node.obj_hash == 0:
            print(node.translation, end=' ')
        elif word_hash < node.obj_hash:
            cls.search_tree(word_hash, node.left_child)
        else:
            cls.search_tree(word_hash, node.right_child)

    @classmethod
    def translate(cls, root):
        cls.tree_root = root
        cls.input = open(GetDataAndBuildTree.get_txt_location(), 'r')
        while True:
            sentence = cls.input.readline().split(" ")
            if sentence == ['']:
                break
            for word in sentence:
                cls.search_tree(cls.hash_of_string(word), cls.tree_root)
        cls.input.close()

# this is the main function which calls get data and build tree
# then depending on if the tree.txt file exists or not two different
# things happen after we have our tree root and tree in the memory we
# can translate


def main():
    root = GetDataAndBuildTree.main()
    Translate.translate(root)



if __name__ == '__main__':
    main()


