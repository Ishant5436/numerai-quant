import random
from typing import List, Tuple, Optional
from chimera.evaluator import ChimeraOpcode, ChimeraInstruction, InstructionBuilder

UNARY_OPS = [
    ChimeraOpcode.TANH,
    ChimeraOpcode.SIGMOID,
    ChimeraOpcode.SIGN,
    ChimeraOpcode.ABS,
    ChimeraOpcode.LOG1P,
    ChimeraOpcode.ZSCORE,
]

BINARY_OPS = [
    ChimeraOpcode.ADD,
    ChimeraOpcode.SUB,
    ChimeraOpcode.MUL,
    ChimeraOpcode.SAFE_DIV,
    ChimeraOpcode.MAX,
    ChimeraOpcode.MIN,
]

OP_NAMES = {
    ChimeraOpcode.ADD: "add",
    ChimeraOpcode.SUB: "sub",
    ChimeraOpcode.MUL: "mul",
    ChimeraOpcode.SAFE_DIV: "safe_div",
    ChimeraOpcode.MAX: "max",
    ChimeraOpcode.MIN: "min",
    ChimeraOpcode.TANH: "tanh",
    ChimeraOpcode.SIGMOID: "sigmoid",
    ChimeraOpcode.SIGN: "sign",
    ChimeraOpcode.ABS: "abs",
    ChimeraOpcode.LOG1P: "log1p",
    ChimeraOpcode.ZSCORE: "zscore",
}

class Node:
    def depth(self) -> int:
        raise NotImplementedError

    def to_formula(self) -> str:
        raise NotImplementedError

    def emit(self, ib: InstructionBuilder, reg_alloc: List[int]) -> int:
        raise NotImplementedError

    def clone(self) -> "Node":
        raise NotImplementedError

    def get_feature_indices(self) -> List[int]:
        raise NotImplementedError

    def compile_to_bytecode(self) -> List[ChimeraInstruction]:
        ib = InstructionBuilder()
        free_regs = list(range(16))  # All 16 registers available
        self.emit(ib, free_regs)
        return ib.build()

class FeatureNode(Node):
    def __init__(self, feat_idx: int):
        self.feat_idx = feat_idx

    def depth(self) -> int:
        return 1

    def to_formula(self) -> str:
        return f"feat_{self.feat_idx}"

    def emit(self, ib: InstructionBuilder, free_regs: List[int]) -> int:
        reg = free_regs.pop(0)
        ib.load_feat(reg=reg, feat_idx=self.feat_idx)
        return reg

    def clone(self) -> "FeatureNode":
        return FeatureNode(self.feat_idx)

    def get_feature_indices(self) -> List[int]:
        return [self.feat_idx]

class ImmNode(Node):
    def __init__(self, val: float):
        self.val = float(val)

    def depth(self) -> int:
        return 1

    def to_formula(self) -> str:
        return f"{self.val:.2f}"

    def emit(self, ib: InstructionBuilder, free_regs: List[int]) -> int:
        reg = free_regs.pop(0)
        ib.load_imm(reg=reg, val=self.val)
        return reg

    def clone(self) -> "ImmNode":
        return ImmNode(self.val)

    def get_feature_indices(self) -> List[int]:
        return []

class UnaryNode(Node):
    def __init__(self, op: ChimeraOpcode, child: Node):
        self.op = op
        self.child = child

    def depth(self) -> int:
        return 1 + self.child.depth()

    def to_formula(self) -> str:
        name = OP_NAMES.get(self.op, "unary")
        return f"{name}({self.child.to_formula()})"

    def emit(self, ib: InstructionBuilder, free_regs: List[int]) -> int:
        child_reg = self.child.emit(ib, free_regs)
        # Unary ops write in-place to child_reg (no new register needed)
        out_reg = child_reg
        if self.op == ChimeraOpcode.TANH:
            ib.tanh(out_reg, child_reg)
        elif self.op == ChimeraOpcode.SIGMOID:
            ib.sigmoid(out_reg, child_reg)
        elif self.op == ChimeraOpcode.SIGN:
            ib.sign(out_reg, child_reg)
        elif self.op == ChimeraOpcode.ABS:
            ib.abs(out_reg, child_reg)
        elif self.op == ChimeraOpcode.LOG1P:
            ib.log1p(out_reg, child_reg)
        elif self.op == ChimeraOpcode.ZSCORE:
            ib.zscore(out_reg, child_reg)
        return out_reg

    def clone(self) -> "UnaryNode":
        return UnaryNode(self.op, self.child.clone())

    def get_feature_indices(self) -> List[int]:
        return self.child.get_feature_indices()

class BinaryNode(Node):
    def __init__(self, op: ChimeraOpcode, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def to_formula(self) -> str:
        name = OP_NAMES.get(self.op, "binary")
        return f"{name}({self.left.to_formula()}, {self.right.to_formula()})"

    def emit(self, ib: InstructionBuilder, free_regs: List[int]) -> int:
        r_left = self.left.emit(ib, free_regs)
        r_right = self.right.emit(ib, free_regs)
        # Write result into r_left, release r_right back to free pool
        out_reg = r_left
        if self.op == ChimeraOpcode.ADD:
            ib.add(out_reg, r_left, r_right)
        elif self.op == ChimeraOpcode.SUB:
            ib.sub(out_reg, r_left, r_right)
        elif self.op == ChimeraOpcode.MUL:
            ib.mul(out_reg, r_left, r_right)
        elif self.op == ChimeraOpcode.SAFE_DIV:
            ib.safe_div(out_reg, r_left, r_right)
        elif self.op == ChimeraOpcode.MAX:
            ib.max(out_reg, r_left, r_right)
        elif self.op == ChimeraOpcode.MIN:
            ib.min(out_reg, r_left, r_right)
        free_regs.insert(0, r_right)
        return out_reg

    def clone(self) -> "BinaryNode":
        return BinaryNode(self.op, self.left.clone(), self.right.clone())

    def get_feature_indices(self) -> List[int]:
        return self.left.get_feature_indices() + self.right.get_feature_indices()

class ASTGenerator:
    def __init__(self, num_features: int, max_depth: int = 4):
        assert num_features > 0, "num_features must be > 0"
        assert max_depth >= 1 and max_depth <= 6, "max_depth must be between 1 and 6"
        self.num_features = num_features
        self.max_depth = max_depth

    def random_leaf(self) -> Node:
        feat_idx = random.randint(0, self.num_features - 1)
        return FeatureNode(feat_idx)

    def random_tree(self, max_depth: Optional[int] = None) -> Node:
        limit = max_depth if max_depth is not None else self.max_depth
        if limit <= 1:
            return self.random_leaf()
        
        # Decide branch or leaf
        choice = random.random()
        if choice < 0.35:
            op = random.choice(UNARY_OPS)
            child = self.random_tree(limit - 1)
            return UnaryNode(op, child)
        else:
            op = random.choice(BINARY_OPS)
            left = self.random_tree(limit - 1)
            right = self.random_tree(limit - 1)
            return BinaryNode(op, left, right)

    def _collect_nodes(self, root: Node) -> List[Tuple[Node, Optional[Node], str]]:
        # Returns list of (node, parent, role)
        res = [(root, None, "root")]
        stack = [(root, None, "root")]
        while stack:
            curr, parent, role = stack.pop()
            if isinstance(curr, UnaryNode):
                res.append((curr.child, curr, "child"))
                stack.append((curr.child, curr, "child"))
            elif isinstance(curr, BinaryNode):
                res.append((curr.left, curr, "left"))
                res.append((curr.right, curr, "right"))
                stack.append((curr.left, curr, "left"))
                stack.append((curr.right, curr, "right"))
        return res

    def mutate(self, tree: Node) -> Node:
        cloned = tree.clone()
        nodes = self._collect_nodes(cloned)
        target, parent, role = random.choice(nodes)

        # Mutate target
        mut_type = random.random()
        if mut_type < 0.4 and isinstance(target, FeatureNode):
            new_target = FeatureNode(random.randint(0, self.num_features - 1))
        elif mut_type < 0.7 and (isinstance(target, UnaryNode) or isinstance(target, BinaryNode)):
            if isinstance(target, UnaryNode):
                new_target = UnaryNode(random.choice(UNARY_OPS), target.child)
            else:
                new_target = BinaryNode(random.choice(BINARY_OPS), target.left, target.right)
        else:
            # Replace subtree with fresh random subtree of appropriate depth
            remaining_depth = max(1, self.max_depth - (tree.depth() - target.depth()))
            new_target = self.random_tree(remaining_depth)

        if parent is None:
            return new_target if new_target.depth() <= self.max_depth else cloned
        
        if role == "child":
            parent.child = new_target
        elif role == "left":
            parent.left = new_target
        elif role == "right":
            parent.right = new_target

        if cloned.depth() > self.max_depth:
            return tree.clone()
        return cloned

    def crossover(self, p1: Node, p2: Node) -> Tuple[Node, Node]:
        c1 = p1.clone()
        c2 = p2.clone()

        nodes1 = self._collect_nodes(c1)
        nodes2 = self._collect_nodes(c2)

        t1, par1, role1 = random.choice(nodes1)
        t2, par2, role2 = random.choice(nodes2)

        # Swap subtrees
        sub1 = t1.clone()
        sub2 = t2.clone()

        if par1 is None:
            c1 = sub2
        else:
            setattr(par1, role1, sub2)

        if par2 is None:
            c2 = sub1
        else:
            setattr(par2, role2, sub1)

        # Enforce max depth constraint
        out1 = c1 if c1.depth() <= self.max_depth else p1.clone()
        out2 = c2 if c2.depth() <= self.max_depth else p2.clone()
        return out1, out2
