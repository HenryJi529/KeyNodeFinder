class Tree:
    class TreeNode:
        def __init__(self, value):
            self.value = value
            self.children = []

        def __str__(self):
            return f"Node({self.value})"

        def add_child(self, child):
            self.children.append(child)

        def remove_child(self, child):
            self.children.remove(child)

    def __init__(self):
        self.root = None

    def is_empty(self):
        return self.root is None

    def add_node(self, value, parent: TreeNode = None):
        node = self.TreeNode(value)

        if parent is None:
            if self.root is not None:
                raise ValueError("Tree already has a root.")
            self.root = node
        else:
            parent.add_child(node)

        return node

    def remove_node(self, node: TreeNode):
        """删除node"""
        if self.root == node:
            self.root = None
        else:
            parent = self.find_parent(node, self.root)
            if parent is not None:
                parent.remove_child(node)
            else:
                raise ValueError("Node not found in the tree.")

    def find_node(self, value, current: TreeNode = None):
        """查找value对应的node"""
        if current is None:
            current = self.root

        if current.value == value:
            return current

        for child in current.children:
            found_node = self.find_node(value, child)
            if found_node is not None:
                return found_node

        return None

    def find_parent(self, node: TreeNode, current: TreeNode = None):
        """查找node的父节点"""
        if node == self.root:
            return None

        if current is None:
            current = self.root

        if node in current.children:
            return current

        for child in current.children:
            parent = self.find_parent(node, child)
            if parent is not None:
                return parent

        return None

    def find_leaves(self, current: TreeNode = None):
        if current is None:
            current = self.root

        leaves = []

        if not current.children:
            return [current]

        for child in current.children:
            leaves.extend(self.find_leaves(child))

        return leaves

    def find_branch(self, node: TreeNode):
        nodeList = [node]
        parent = self.find_parent(node)
        if parent:
            nodeList.extend(self.find_branch(parent))
        return nodeList

    def remove_branch(self, node: TreeNode):
        parent = self.find_parent(node)
        if parent:
            self.remove_node(node)
            if parent.children:
                pass
            else:
                self.remove_branch(parent)

    def get_leaf_depth(self, leafNode: TreeNode):
        depth = 1
        parent = leafNode
        while True:
            parent = self.find_parent(parent)
            if parent:
                depth += 1
            else:
                break
        return depth

    def depth_first_traversal(self, current: TreeNode = None):
        if current is None:
            current = self.root

        print(current.value, end=" ")

        for child in current.children:
            self.depth_first_traversal(child)

    def breadth_first_traversal(self):
        if self.root is None:
            return

        queue = [self.root]

        while queue:
            node = queue.pop(0)
            print(node.value, end=" ")

            for child in node.children:
                queue.append(child)


if __name__ == "__main__":
    # 创建一个树对象
    tree = Tree()

    # 添加节点
    root = tree.add_node("A")
    node_b = tree.add_node("B", parent=root)
    node_c = tree.add_node("C", parent=root)
    node_d = tree.add_node("D", parent=node_b)
    node_e = tree.add_node("E", parent=node_b)
    node_f = tree.add_node("F", parent=node_c)
    node_g = tree.add_node("G", parent=node_c)
    node_h = tree.add_node("H", parent=root)

    # 判断树是否为空
    print("Is tree empty?", tree.is_empty())

    # 查找叶子
    print("All the leaves:", [str(node) for node in tree.find_leaves()])
    print("All the leaves of C:", [str(node) for node in tree.find_leaves(node_c)])
    print("Find branch of G:", [str(node) for node in tree.find_branch(node_g)])

    # 查找节点
    found_node = tree.find_node("D")
    print("Found node:", found_node.value)

    # 删除节点
    tree.remove_node(node_b)

    # 深度优先遍历
    print("Depth-first traversal:")
    tree.depth_first_traversal()
    print()
    # 广度优先遍历
    print("Breadth-first traversal:")
    tree.breadth_first_traversal()
    print()

    # 删除branch
    tree.remove_branch(node_f)
    tree.breadth_first_traversal()  # NOTE: 此时应该打印 [A C H G]
    print()

    node_i = tree.add_node("I", parent=node_h)
    node_j = tree.add_node("J", parent=node_i)
    print("Breadth-first traversal:")
    tree.breadth_first_traversal()  # NOTE: 此时应该打印 [A C H G I J]
    print()

    tree.remove_branch(node_i)
    print("Breadth-first traversal:")
    tree.breadth_first_traversal()  # NOTE: 此时应该打印 [A C G]
    print()

    print(f"A深度为: {tree.get_leaf_depth(root)}")
    print(f"C深度为: {tree.get_leaf_depth(node_c)}")
    print(f"G深度为: {tree.get_leaf_depth(node_g)}")
