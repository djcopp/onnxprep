"""Tiny graph builder over ``onnx.helper``: no dependencies beyond onnx + numpy.

Every recipe in onnxprep is built from *standard* ``ai.onnx`` operators only, so the resulting models
run on any ONNX runtime / execution provider without custom-op libraries.
"""

from __future__ import annotations

import itertools

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17  # first opset with STFT / DFT / MelWeightMatrix; widely supported by runtimes

_DTYPES = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float16): TensorProto.FLOAT16,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.uint8): TensorProto.UINT8,
    np.dtype(np.bool_): TensorProto.BOOL,
}


class Graph:
    """Minimal SSA-style builder. ``g.op("Add", a, b)`` returns the output tensor name."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        self._ids = itertools.count()

    def _fresh(self, hint: str) -> str:
        return f"{hint}_{next(self._ids)}"

    def input(self, name: str, dtype, shape) -> str:
        self.inputs.append(helper.make_tensor_value_info(name, _DTYPES[np.dtype(dtype)], shape))
        return name

    def output(self, tensor: str, name: str, dtype, shape) -> str:
        self.op("Identity", tensor, out=name)
        self.outputs.append(helper.make_tensor_value_info(name, _DTYPES[np.dtype(dtype)], shape))
        return name

    def const(self, value, name: str | None = None, dtype=None) -> str:
        arr = np.asarray(value, dtype=dtype)
        name = name or self._fresh("c")
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def op(self, op_type: str, *inputs: str, out: str | None = None, n_out: int = 1, **attrs):
        outs = [out] if out else [self._fresh(op_type.lower()) for _ in range(n_out)]
        self.nodes.append(helper.make_node(op_type, list(inputs), outs, **attrs))
        return outs[0] if n_out == 1 else outs

    def model(self, doc: str = "", metadata: dict[str, str] | None = None) -> onnx.ModelProto:
        g = helper.make_graph(self.nodes, self.name, self.inputs, self.outputs, initializer=self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="onnxprep")
        m.doc_string = doc
        for k, v in (metadata or {}).items():
            m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=str(v)))
        onnx.checker.check_model(m, full_check=True)
        return m
