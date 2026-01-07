from typing import Dict, Set, List, Tuple, Union
import types
import bytecode as bc
from bytecode import Instr, BasicBlock, ControlFlowGraph, Bytecode

_USE_INSTRS = (
    "LOAD_NAME",
    "LOAD_FAST",
    "LOAD_FAST_CHECK",
    "LOAD_FAST_AND_CLEAR",
    "LOAD_FAST_BORROW",
)
_SUPER_USE_INSTRS = ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW")
_SUPER_DEF_INSTRS = "STORE_FAST_STORE_FAST"
_DEF_INSTRS = ("STORE_NAME", "STORE_FAST")


class LivenessAnalysis:
    def __init__(self, code: Union[types.CodeType, Bytecode, ControlFlowGraph]):
        """Initialize liveness analysis for Python code.

        Args:
            code: Can be a code object, Bytecode, or ControlFlowGraph
        """
        if isinstance(code, types.CodeType):
            self.bytecode = bc.Bytecode.from_code(code)
            self.cfg = bc.ControlFlowGraph.from_bytecode(self.bytecode)
        elif isinstance(code, bc.Bytecode):
            self.bytecode = code
            self.cfg = bc.ControlFlowGraph.from_bytecode(code)
        elif isinstance(code, bc.ControlFlowGraph):
            self.cfg = code
        else:
            raise TypeError(
                f"Expected code, Bytecode, or ControlFlowGraph, got {type(code)}"
            )

        # Maps from block index to sets of live variables at entry and exit
        self.block_live_in: Dict[int, Set[str]] = {}
        self.block_live_out: Dict[int, Set[str]] = {}

        # Maps from instruction offset to live variables
        self.offset_to_live_vars: Dict[int, Set[str]] = {}
        self._instr_offset_set, self._instr_offsets = self._get_instr_offsets()
        self._localvars = set()

        # Perform the analysis
        self._analyze()

    def _get_instr_offsets(self):
        offsets = []
        for instr in self.bytecode:
            if isinstance(instr, Instr):
                offsets.append(instr.offset)
        return set(offsets), offsets

    def _is_valid_offset(self, offset: int) -> bool:
        return offset in self._instr_offset_set

    def _is_use(self, instr: Instr) -> bool:
        return instr.name in _USE_INSTRS

    def _is_def(self, instr: Instr) -> bool:
        return instr.name in _DEF_INSTRS

    def _is_super_use(self, instr: Instr) -> bool:
        return instr.name in _SUPER_USE_INSTRS

    def _is_super_def(self, instr: Instr) -> bool:
        return instr.name in _SUPER_DEF_INSTRS

    def _get_uses_and_defs(self, block: BasicBlock) -> Tuple[Set[str], Set[str]]:
        """Get the variables used and defined in a basic block.

        Returns:
            Tuple of (used_vars, defined_vars)
        """
        used_vars = set()
        defined_vars = set()

        for instr in block:
            if not isinstance(instr, Instr):
                continue

            if self._is_use(instr):
                if isinstance(instr.arg, str):
                    used_vars.add(instr.arg)
            elif self._is_def(instr):
                if isinstance(instr.arg, str):
                    defined_vars.add(instr.arg)
            # special 'super-instruction'
            elif self._is_super_use(instr):
                arg0, arg1 = instr.arg
                if isinstance(arg0, str):
                    used_vars.add(arg0)
                if isinstance(arg1, str):
                    used_vars.add(arg1)
            elif self._is_super_def(instr):
                arg0, arg1 = instr.arg
                if isinstance(arg0, str):
                    defined_vars.add(arg0)
                if isinstance(arg1, str):
                    defined_vars.add(arg1)
            elif instr.name == "STORE_FAST_LOAD_FAST":
                store_var, load_var = instr.arg
                if isinstance(store_var, str):
                    defined_vars.add(store_var)
                if isinstance(load_var, str):
                    used_vars.add(load_var)

        return used_vars, defined_vars

    def _analyze(self):
        """Perform liveness analysis on the CFG."""
        # Use block indices instead of block objects as dictionary keys
        block_indices = {id(block): i for i, block in enumerate(self.cfg)}

        # Initialize live_in and live_out sets for all blocks
        self.block_live_in = {i: set() for i in block_indices.values()}
        self.block_live_out = {i: set() for i in block_indices.values()}

        # Mapping from block ID to block object for convenience
        self.blocks_by_id = {id(block): block for block in self.cfg}

        # Iteratively compute liveness until fixed point
        changed = True
        while changed:
            changed = False

            # Process blocks in reverse order (backward analysis)
            for block in reversed(list(self.cfg)):
                block_idx = block_indices[id(block)]

                # Compute new live_out as the union of live_in of all successors
                new_live_out = set()

                # Add live_in from next_block if it exists
                if block.next_block:
                    next_block_idx = block_indices[id(block.next_block)]
                    new_live_out.update(self.block_live_in[next_block_idx])

                # Add live_in from jump targets
                last_instr = None
                for instr in reversed(block):
                    if isinstance(instr, Instr):
                        last_instr = instr
                        break

                if last_instr and last_instr.has_jump():
                    if isinstance(last_instr.arg, BasicBlock):
                        target_idx = block_indices[id(last_instr.arg)]
                        new_live_out.update(self.block_live_in[target_idx])

                # Check if live_out changed
                if new_live_out != self.block_live_out[block_idx]:
                    changed = True
                    self.block_live_out[block_idx] = new_live_out

                # Compute new live_in
                uses, defs = self._get_uses_and_defs(block)
                new_live_in = uses.union(self.block_live_out[block_idx] - defs)

                # Check if live_in changed
                if new_live_in != self.block_live_in[block_idx]:
                    changed = True
                    self.block_live_in[block_idx] = new_live_in

        # Store the block_indices for later use
        self.block_indices = block_indices

        # Compute liveness for each instruction within blocks
        self._compute_instruction_liveness(block_indices)

    def _compute_instruction_liveness(self, block_indices):
        """Compute liveness information for each instruction within blocks."""
        # Create a dictionary to map (block_idx, instr_idx) to live variables
        self.instr_to_live_vars = {}

        for block in self.cfg:
            block_idx = block_indices[id(block)]
            # Start with live variables at block exit
            live_vars = self.block_live_out[block_idx].copy()

            # Process instructions in reverse order
            for i in range(len(block) - 1, -1, -1):
                instr = block[i]
                if isinstance(instr, Instr):
                    # Record live variables at this instruction
                    if instr.offset is not None:
                        self.offset_to_live_vars[instr.offset] = live_vars.copy()

                    # Also store by block and instruction index
                    self.instr_to_live_vars[(block_idx, i)] = live_vars.copy()

                    # Update live variables based on this instruction
                    if self._is_def(instr):
                        if isinstance(instr.arg, str):
                            live_vars.discard(instr.arg)
                            self._localvars.add(instr.arg)
                    elif self._is_use(instr):
                        if isinstance(instr.arg, str):
                            live_vars.add(instr.arg)
                            self._localvars.add(instr.arg)
                    elif self._is_super_use(instr):
                        arg0, arg1 = instr.arg
                        if isinstance(arg0, str):
                            live_vars.add(arg0)
                            self._localvars.add(arg0)
                        if isinstance(arg1, str):
                            live_vars.add(arg1)
                            self._localvars.add(arg1)
                    elif self._is_super_def(instr):
                        arg0, arg1 = instr.arg
                        if isinstance(arg0, str):
                            live_vars.add(arg0)
                            self._localvars.add(arg0)
                        if isinstance(arg1, str):
                            live_vars.add(arg1)
                    elif instr.name == "STORE_FAST_LOAD_FAST":
                        store_var, load_var = instr.arg
                        if isinstance(store_var, str):
                            live_vars.discard(store_var)
                            self._localvars.add(store_var)
                        if isinstance(load_var, str):
                            live_vars.add(load_var)
                            self._localvars.add(load_var)

    def get_live_variables_at_offset(self, offset: int) -> Set[str]:
        """Get the set of live variables at a given bytecode offset."""
        if self._is_valid_offset(offset):
            if offset in self.offset_to_live_vars:
                return self.offset_to_live_vars[offset]
            return set()
        raise ValueError(f"Invalid offset: {offset}")

    def get_dead_variables_at_offset(self, offset: int) -> Set[str]:
        """Get the set of dead variables at a given bytecode offset."""
        live_vars = self.get_live_variables_at_offset(offset)
        return self._localvars - live_vars

    def get_offsets(self) -> List[int]:
        """Get the list of instruction offsets."""
        return self._instr_offsets


liveness_cache = {}


def get_dead_variables_at_offset(code: types.CodeType, offset: int) -> Set[str]:
    """Get the set of dead variables at a given bytecode offset."""
    h = hash(code.co_name)
    if h not in liveness_cache:
        liveness_cache[h] = LivenessAnalysis(code)
    return liveness_cache[h].get_dead_variables_at_offset(offset)
