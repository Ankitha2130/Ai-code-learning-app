import ast
from graphviz import Digraph
import uuid

class FlowchartGenerator(ast.NodeVisitor):
    def __init__(self):
        self.graph = Digraph(format='png')
        self.counter = 0
        self.stack = []
        self.last_node = None

    def new_node(self, label):
        node_id = str(uuid.uuid4())[:8]
        self.graph.node(node_id, label)
        return node_id

    def connect(self, from_id, to_id, label=''):
        self.graph.edge(from_id, to_id, label)

    def visit_FunctionDef(self, node):
        start_id = self.new_node(f'Function: {node.name}')
        self.last_node = start_id
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node):
        cond_id = self.new_node(f'If: {ast.unparse(node.test)}')
        self.connect(self.last_node, cond_id)
        self.stack.append(self.last_node)
        self.last_node = cond_id
        for stmt in node.body:
            self.visit(stmt)
        if node.orelse:
            else_id = self.new_node('Else')
            self.connect(cond_id, else_id, label='False')
            self.last_node = else_id
            for stmt in node.orelse:
                self.visit(stmt)
        self.last_node = self.stack.pop()

    def visit_While(self, node):
        loop_id = self.new_node(f'While: {ast.unparse(node.test)}')
        self.connect(self.last_node, loop_id)
        self.stack.append(loop_id)
        self.last_node = loop_id
        for stmt in node.body:
            self.visit(stmt)
        self.connect(self.last_node, loop_id)  # loop back
        self.last_node = self.stack.pop()

    def visit_For(self, node):
        loop_id = self.new_node(f'For: {ast.unparse(node.target)} in {ast.unparse(node.iter)}')
        self.connect(self.last_node, loop_id)
        self.stack.append(loop_id)
        self.last_node = loop_id
        for stmt in node.body:
            self.visit(stmt)
        self.connect(self.last_node, loop_id)  # loop back
        self.last_node = self.stack.pop()

    def visit_Return(self, node):
        return_id = self.new_node(f'Return: {ast.unparse(node.value)}')
        self.connect(self.last_node, return_id)
        self.last_node = return_id

    def visit_Assign(self, node):
        assign_id = self.new_node(f'Assign: {ast.unparse(node)}')
        self.connect(self.last_node, assign_id)
        self.last_node = assign_id

    def visit_Expr(self, node):
        expr_id = self.new_node(f'Expr: {ast.unparse(node)}')
        self.connect(self.last_node, expr_id)
        self.last_node = expr_id

def generate_flowchart(code, output_path='flowchart.png'):
    tree = ast.parse(code)
    generator = FlowchartGenerator()
    generator.visit(tree)
    generator.graph.render(output_path, cleanup=True)
    return output_path + ".png"
