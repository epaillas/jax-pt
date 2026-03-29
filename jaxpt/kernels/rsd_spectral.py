from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from sympy import lambdify, symbols
from sympy.parsing.mathematica import parse_mathematica

from ..config import PTSettings
from ..cosmology import LinearPowerInput, NativeFFTLogInput, prepare_native_fftlog_input
from .spectral import (
    _analytic_realspace_kernel_registry,
    _fftlog_coefficients_jax,
    _interpolate_to_output_jax,
    _j_np,
    _quadratic_form_columns,
)

_TRANSFER_BIAS = -0.8
_TRANSFER_BIAS_BIASED = -1.25

_MATTER_MULTIPLIER_TEXT = {
    "vv0_f2": r"""-1/140*(Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((-120 + 145*n2 + 448*n1^4*(-1 + n2)*n2 + 948*n2^2 - 1892*n2^3 + 1152*n2^4 - 224*n2^5 + 12*n1^3*(7 + 146*n2 - 272*n2^2 + 112*n2^3) + 4*n1^2*(-69 - 540*n2 + 1652*n2^2 - 1352*n2^3 + 336*n2^4) + n1*(321 + 792*n2 - 4840*n2^2 + 6008*n2^3 - 2816*n2^4 + 448*n2^5))*Sin[2*n1*Pi] + (224*n1^5*(-1 + 2*n2) + 64*n1^4*(18 - 44*n2 + 21*n2^2) + 3*(-40 + 107*n2 - 92*n2^2 + 28*n2^3) + 4*n1^3*(-473 + 1502*n2 - 1352*n2^2 + 336*n2^3) + n1*(145 + 792*n2 - 2160*n2^2 + 1752*n2^3 - 448*n2^4) + 4*n1^2*(237 - 1210*n2 + 1652*n2^2 - 816*n2^3 + 112*n2^4))*Sin[2*n2*Pi] + (120 - 321*n2 + 448*n1^4*(-1 + n2)*n2 + 276*n2^2 - 84*n2^3 + 4*n1^3*(-21 + 466*n2 - 648*n2^2 + 224*n2^3) + n1*(-321 + 1856*n2 - 2864*n2^2 + 1864*n2^3 - 448*n2^4) + 4*n1^2*(69 - 716*n2 + 1156*n2^2 - 648*n2^3 + 112*n2^4))*Sin[2*(n1 + n2)*Pi]))/(n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "vv0_f3": r"""-1/140*(Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((-90 + 99*n2 + 320*n1^4*(-1 + n2)*n2 + 716*n2^2 - 1388*n2^3 + 832*n2^4 - 160*n2^5 + 4*n1^3*(15 + 322*n2 - 592*n2^2 + 240*n2^3) + 4*n1^2*(-51 - 412*n2 + 1228*n2^2 - 984*n2^3 + 240*n2^4) + n1*(243 + 632*n2 - 3672*n2^2 + 4456*n2^3 - 2048*n2^4 + 320*n2^5))*Sin[2*n1*Pi] + (160*n1^5*(-1 + 2*n2) + 64*n1^4*(13 - 32*n2 + 15*n2^2) + 3*(-30 + 81*n2 - 68*n2^2 + 20*n2^3) + 4*n1^3*(-347 + 1114*n2 - 984*n2^2 + 240*n2^3) + n1*(99 + 632*n2 - 1648*n2^2 + 1288*n2^3 - 320*n2^4) + 4*n1^2*(179 - 918*n2 + 1228*n2^2 - 592*n2^3 + 80*n2^4))*Sin[2*n2*Pi] + (90 - 243*n2 + 320*n1^4*(-1 + n2)*n2 + 204*n2^2 - 60*n2^3 + 4*n1^3*(-15 + 342*n2 - 472*n2^2 + 160*n2^3) + n1*(-243 + 1424*n2 - 2160*n2^2 + 1368*n2^3 - 320*n2^4) + 4*n1^2*(51 - 540*n2 + 860*n2^2 - 472*n2^3 + 80*n2^4))*Sin[2*(n1 + n2)*Pi]))/(n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "vd0_f1": r"""-1/42*(Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((-90 + 177*n2 + 448*n1^4*(-1 + n2)*n2 + 676*n2^2 - 1636*n2^3 + 1088*n2^4 - 224*n2^5 + 4*n1^3*(21 + 374*n2 - 752*n2^2 + 336*n2^3) + 4*n1^2*(-57 - 356*n2 + 1316*n2^2 - 1224*n2^3 + 336*n2^4) + n1*(225 + 328*n2 - 3336*n2^2 + 4856*n2^3 - 2560*n2^4 + 448*n2^5))*Sin[2*n1*Pi] + (224*n1^5*(-1 + 2*n2) + 64*n1^4*(17 - 40*n2 + 21*n2^2) + 3*(-30 + 75*n2 - 76*n2^2 + 28*n2^3) + 4*n1^3*(-409 + 1214*n2 - 1224*n2^2 + 336*n2^3) + n1*(177 + 328*n2 - 1424*n2^2 + 1496*n2^3 - 448*n2^4) + 4*n1^2*(169 - 834*n2 + 1316*n2^2 - 752*n2^3 + 112*n2^4))*Sin[2*n2*Pi] + (90 - 225*n2 + 448*n1^4*(-1 + n2)*n2 + 228*n2^2 - 84*n2^3 + 4*n1^3*(-21 + 402*n2 - 584*n2^2 + 224*n2^3) + n1*(-225 + 1168*n2 - 2064*n2^2 + 1608*n2^3 - 448*n2^4) + 4*n1^2*(57 - 516*n2 + 916*n2^2 - 584*n2^3 + 112*n2^4))*Sin[2*(n1 + n2)*Pi]))/(n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "vd0_f2": r"""-1/30*(Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((-30 + 85*n2 + 192*n1^4*(-1 + n2)*n2 + 212*n2^2 - 628*n2^3 + 448*n2^4 - 96*n2^5 + 4*n1^3*(9 + 142*n2 - 304*n2^2 + 144*n2^3) + 4*n1^2*(-21 - 100*n2 + 468*n2^2 - 488*n2^3 + 144*n2^4) + n1*(69 + 8*n2 - 1000*n2^2 + 1752*n2^3 - 1024*n2^4 + 192*n2^5))*Sin[2*n1*Pi] + (-30 + 69*n2 - 84*n2^2 + 36*n2^3 + 96*n1^5*(-1 + 2*n2) + 64*n1^4*(7 - 16*n2 + 9*n2^2) + 4*n1^3*(-157 + 438*n2 - 488*n2^2 + 144*n2^3) + n1*(85 + 8*n2 - 400*n2^2 + 568*n2^3 - 192*n2^4) + 4*n1^2*(53 - 250*n2 + 468*n2^2 - 304*n2^3 + 48*n2^4))*Sin[2*n2*Pi] + (30 - 69*n2 + 192*n1^4*(-1 + n2)*n2 + 84*n2^2 - 36*n2^3 + 4*n1^3*(-9 + 154*n2 - 232*n2^2 + 96*n2^3) + n1*(-69 + 304*n2 - 656*n2^2 + 616*n2^3 - 192*n2^4) + 4*n1^2*(21 - 164*n2 + 324*n2^2 - 232*n2^3 + 48*n2^4))*Sin[2*(n1 + n2)*Pi]))/(n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "dd0_f0": r"""(Csc[n1*Pi]*Csc[n2*Pi]*(-2*(60 + 448*n1^4*(-1 + n2)*n2 - 3*n2*(43 + 4*n2*(-15 + 7*n2)) + 4*n1^3*(-21 + 2*n2*(169 + 4*n2*(-65 + 28*n2))) + 4*n1^2*(45 + 4*n2*(-79 + n2*(169 + 2*n2*(-65 + 14*n2)))) + n1*(-129 - 8*n2*(-60 + n2*(158 + n2*(-169 + 56*n2)))))*Cos[(n1 + n2)*Pi] - Csc[(n1 + n2)*Pi]*((-60 + 3*n1*(43 + 4*n1*(-15 + 7*n1)) + 209*n2 - 8*n1*(17 + (-2 + n1)*n1*(-43 + 56*n1))*n2 + 4*(101 + 2*n1*(-229 + 2*n1*(245 + 4*n1*(-43 + 7*n1))))*n2^2 + 4*(-345 + 2*n1*(463 + 4*n1*(-137 + 42*n1)))*n2^3 + 64*(16 + 3*n1*(-12 + 7*n1))*n2^4 + 224*(-1 + 2*n1)*n2^5)*Sin[2*n1*Pi] + (-60 + n1*(209 + 4*n1*(101 + n1*(-345 + 8*(32 - 7*n1)*n1))) + 129*n2 + 8*n1*(-17 + n1*(-229 + n1*(463 + 8*n1*(-36 + 7*n1))))*n2 + 4*(-45 + 4*n1*(-43 + n1*(245 - 274*n1 + 84*n1^2)))*n2^2 + 4*(21 + 2*n1*(155 + 8*n1*(-43 + 21*n1)))*n2^3 + 448*(-1 + n1)*n1*n2^4)*Sin[2*n2*Pi])))/(28*n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "dd0_f1": r"""-1/12*(Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((64*n1^4*(-1 + n2)*n2 + 4*n1^3*(3 + 26*n2 - 80*n2^2 + 48*n2^3) + n2*(39 - 20*n2 - 124*n2^2 + 128*n2^3 - 32*n2^4) + 4*n1^2*(-3 + 28*n2 + 44*n2^2 - 120*n2^3 + 48*n2^4) + n1*(-9 - 152*n2 + 168*n2^2 + 200*n2^3 - 256*n2^4 + 64*n2^5))*Sin[2*n1*Pi] + (32*n1^5*(-1 + 2*n2) + 64*n1^4*(2 - 4*n2 + 3*n2^2) + 3*n2*(-3 - 4*n2 + 4*n2^2) + 4*n1^3*(-31 + 50*n2 - 120*n2^2 + 48*n2^3) + n1*(39 - 152*n2 + 112*n2^2 + 104*n2^3 - 64*n2^4) + 4*n1^2*(-5 + 42*n2 + 44*n2^2 - 80*n2^3 + 16*n2^4))*Sin[2*n2*Pi] + (64*n1^4*(-1 + n2)*n2 + 3*n2*(3 + 4*n2 - 4*n2^2) + 4*n1^3*(-3 + 30*n2 - 56*n2^2 + 32*n2^3) + n1*(9 - 128*n2 + 48*n2^2 + 120*n2^3 - 64*n2^4) + 4*n1^2*(3 + 12*n2 + 28*n2^2 - 56*n2^3 + 16*n2^4))*Sin[2*(n1 + n2)*Pi]))/(n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "vv2_f3": r"""-1/84*(Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((-165 + 214*n2 + 640*n1^4*(-1 + n2)*n2 + 1296*n2^2 - 2648*n2^3 + 1632*n2^4 - 320*n2^5 + 24*n1^3*(5 + 102*n2 - 192*n2^2 + 80*n2^3) + 16*n1^2*(-24 - 183*n2 + 572*n2^2 - 476*n2^3 + 120*n2^4) + 2*n1*(219 + 516*n2 - 3296*n2^2 + 4168*n2^3 - 1984*n2^4 + 320*n2^5))*Sin[2*n1*Pi] + (320*n1^5*(-1 + 2*n2) + 32*n1^4*(51 - 124*n2 + 60*n2^2) + 3*(-55 + 146*n2 - 128*n2^2 + 40*n2^3) + 8*n1^3*(-331 + 1042*n2 - 952*n2^2 + 240*n2^3) + n1*(214 + 1032*n2 - 2928*n2^2 + 2448*n2^3 - 640*n2^4) + 16*n1^2*(81 - 412*n2 + 572*n2^2 - 288*n2^3 + 40*n2^4))*Sin[2*n2*Pi] + (165 - 438*n2 + 640*n1^4*(-1 + n2)*n2 + 384*n2^2 - 120*n2^3 + 8*n1^3*(-15 + 326*n2 - 456*n2^2 + 160*n2^3) + n1*(-438 + 2504*n2 - 3920*n2^2 + 2608*n2^3 - 640*n2^4) + 16*n1^2*(24 - 245*n2 + 400*n2^2 - 228*n2^3 + 40*n2^4))*Sin[2*(n1 + n2)*Pi]))/(n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "vd2_f2": r"""-1/84*(Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((-195 + 728*n2 + 1536*n1^4*(-1 + n2)*n2 + 1288*n2^2 - 4640*n2^3 + 3488*n2^4 - 768*n2^5 + 32*n1^3*(9 + 130*n2 - 292*n2^2 + 144*n2^3) + 8*n1^2*(-75 - 262*n2 + 1620*n2^2 - 1856*n2^3 + 576*n2^4) + 8*n1*(51 - 79*n2 - 718*n2^2 + 1536*n2^3 - 976*n2^4 + 192*n2^5))*Sin[2*n1*Pi] + (768*n1^5*(-1 + 2*n2) + 32*n1^4*(109 - 244*n2 + 144*n2^2) + 3*(-65 + 136*n2 - 200*n2^2 + 96*n2^3) + 32*n1^3*(-145 + 384*n2 - 464*n2^2 + 144*n2^3) + 8*n1^2*(161 - 718*n2 + 1620*n2^2 - 1168*n2^3 + 192*n2^4) - 8*n1*(-91 + 79*n2 + 262*n2^2 - 520*n2^3 + 192*n2^4))*Sin[2*n2*Pi] + (195 - 408*n2 + 1536*n1^4*(-1 + n2)*n2 + 600*n2^2 - 288*n2^3 + 32*n1^3*(-9 + 142*n2 - 220*n2^2 + 96*n2^3) + 8*n1^2*(75 - 506*n2 + 1116*n2^2 - 880*n2^3 + 192*n2^4) - 8*n1*(51 - 175*n2 + 506*n2^2 - 568*n2^3 + 192*n2^4))*Sin[2*(n1 + n2)*Pi]))/(n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "vv4_f3": r"""(-2*Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((-195 + 377*n2 + 960*n1^4*(-1 + n2)*n2 + 1468*n2^2 - 3524*n2^3 + 2336*n2^4 - 480*n2^5 + 4*n1^3*(45 + 806*n2 - 1616*n2^2 + 720*n2^3) + 4*n1^2*(-123 - 776*n2 + 2844*n2^2 - 2632*n2^3 + 720*n2^4) + n1*(489 + 736*n2 - 7256*n2^2 + 10488*n2^3 - 5504*n2^4 + 960*n2^5))*Sin[2*n1*Pi] + (480*n1^5*(-1 + 2*n2) + 32*n1^4*(73 - 172*n2 + 90*n2^2) + 3*(-65 + 163*n2 - 164*n2^2 + 60*n2^3) + 4*n1^3*(-881 + 2622*n2 - 2632*n2^2 + 720*n2^3) + n1*(377 + 736*n2 - 3104*n2^2 + 3224*n2^3 - 960*n2^4) + 4*n1^2*(367 - 1814*n2 + 2844*n2^2 - 1616*n2^3 + 240*n2^4))*Sin[2*n2*Pi] + (195 - 489*n2 + 960*n1^4*(-1 + n2)*n2 + 492*n2^2 - 180*n2^3 + 4*n1^3*(-45 + 866*n2 - 1256*n2^2 + 480*n2^3) + n1*(-489 + 2552*n2 - 4480*n2^2 + 3464*n2^3 - 960*n2^4) + n1^2*(492 - 4480*n2 + 7920*n2^2 - 5024*n2^3 + 960*n2^4))*Sin[2*(n1 + n2)*Pi]))/(385*n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
    "vd4_f2": r"""-1/35*(Csc[n1*Pi]*Csc[n2*Pi]*Csc[(n1 + n2)*Pi]*((-15 + 140*n2 + 256*n1^4*(-1 + n2)*n2 + 56*n2^2 - 624*n2^3 + 544*n2^4 - 128*n2^5 + 16*n1^3*(3 + 34*n2 - 88*n2^2 + 48*n2^3) + 8*n1^2*(-9 + 10*n2 + 172*n2^2 - 272*n2^3 + 96*n2^4) + 4*n1*(3 - 94*n2 - 20*n2^2 + 344*n2^3 - 288*n2^4 + 64*n2^5))*Sin[2*n1*Pi] + (128*n1^5*(-1 + 2*n2) + 32*n1^4*(17 - 36*n2 + 24*n2^2) + 3*(-5 + 4*n2 - 24*n2^2 + 16*n2^3) + 16*n1^3*(-39 + 86*n2 - 136*n2^2 + 48*n2^3) + 8*n1^2*(7 - 10*n2 + 172*n2^2 - 176*n2^3 + 32*n2^4) - 4*n1*(-35 + 94*n2 - 20*n2^2 - 136*n2^3 + 64*n2^4))*Sin[2*n2*Pi] + (15 - 12*n2 + 256*n1^4*(-1 + n2)*n2 + 72*n2^2 - 48*n2^3 + 16*n1^3*(-3 + 38*n2 - 64*n2^2 + 32*n2^3) + 8*n1^2*(9 - 26*n2 + 116*n2^2 - 128*n2^3 + 32*n2^4) - 4*n1*(3 + 42*n2 + 52*n2^2 - 152*n2^3 + 64*n2^4))*Sin[2*(n1 + n2)*Pi]))/(n1*(-1 + 4*n1^2)*n2*(-5 + 2*n1 + 2*n2)*(-1 + 4*n2^2))""",
}

_BIAS_MULTIPLIER_TEXT = {
    "b2_vv0_f1": r"""-1/3*(Csc[nu1*Pi]*Csc[nu2*Pi]*Csc[(nu1 + nu2)*Pi]*((15 + 8*nu1*(-1 + nu2) - 24*nu2 + 8*nu2^2)*Sin[2*nu1*Pi] + (15 + 8*nu1^2 + 8*nu1*(-3 + nu2) - 8*nu2)*Sin[2*nu2*Pi] + (9 + 8*nu1*(-1 + nu2) - 8*nu2)*Sin[2*(nu1 + nu2)*Pi]))/((-1 + 2*nu1)*(-1 + 2*nu2)*(-5 + 2*nu1 + 2*nu2))""",
    "bG2_vv0_f1": r"""(Csc[nu1*Pi]*Csc[nu2*Pi]*Csc[(nu1 + nu2)*Pi]*((15 + 16*nu2 + 128*nu1^3*(-1 + nu2)*nu2 - 136*nu2^2 + 128*nu2^3 - 32*nu2^4 + 8*nu1^2*(3 + 46*nu2 - 84*nu2^2 + 32*nu2^3) + 8*nu1*(-6 - 29*nu2 + 94*nu2^2 - 72*nu2^3 + 16*nu2^4))*Sin[2*nu1*Pi] + (32*nu1^4*(-1 + 4*nu2) + 64*nu1^3*(2 - 9*nu2 + 4*nu2^2) + 3*(5 - 16*nu2 + 8*nu2^2) + 8*nu1^2*(-17 + 94*nu2 - 84*nu2^2 + 16*nu2^3) - 8*nu1*(-2 + 29*nu2 - 46*nu2^2 + 16*nu2^3))*Sin[2*nu2*Pi] + (128*nu1^3*(-1 + nu2)*nu2 - 3*(5 - 16*nu2 + 8*nu2^2) + 8*nu1^2*(-3 + 50*nu2 - 60*nu2^2 + 16*nu2^3) - 8*nu1*(-6 + 43*nu2 - 50*nu2^2 + 16*nu2^3))*Sin[2*(nu1 + nu2)*Pi]))/(6*nu1*(-1 + 4*nu1^2)*nu2*(-5 + 2*nu1 + 2*nu2)*(-1 + 4*nu2^2))""",
}


@lru_cache(maxsize=32)
def _compile_multiplier(expression: str, kind: str):
    if kind == "matter":
        expression = expression.replace("n1", "nu1").replace("n2", "nu2")
    nu1, nu2 = symbols("nu1 nu2")
    return lambdify((nu1, nu2), parse_mathematica(expression), modules="numpy")


def _build_transfer_matrices(etam: np.ndarray, expression_map: dict[str, str], kind: str) -> dict[str, np.ndarray]:
    nu_row = -0.5 * etam[:, None]
    nu_col = -0.5 * etam[None, :]
    jmat = _j_np(nu_row, nu_col)
    return {
        name: jmat * np.asarray(_compile_multiplier(expression, kind)(nu_row, nu_col), dtype=complex)
        for name, expression in expression_map.items()
    }


@lru_cache(maxsize=8)
def _analytic_rsd_transfer_registry(nmax: int, k0_over_h: float, kmax_over_h: float) -> dict[str, np.ndarray]:
    delta = np.log(kmax_over_h / k0_over_h) / (nmax - 1.0)
    js = np.arange(nmax + 1, dtype=float) - nmax / 2.0
    etam_transfer = _TRANSFER_BIAS + 2.0j * np.pi * js / (nmax * delta)
    etam2_transfer = _TRANSFER_BIAS_BIASED + 2.0j * np.pi * js / (nmax * delta)
    registry: dict[str, np.ndarray] = {
        "etam_transfer": etam_transfer,
        "etam2_transfer": etam2_transfer,
    }
    registry.update(_build_transfer_matrices(etam_transfer, _MATTER_MULTIPLIER_TEXT, "matter"))
    registry.update(_build_transfer_matrices(etam2_transfer, _BIAS_MULTIPLIER_TEXT, "bias"))
    return registry


def _mu_to_multipoles(mu0, mu2, mu4, mu6, mu8):
    l0 = mu0 + mu2 / 3.0 + mu4 / 5.0 + mu6 / 7.0 + mu8 / 9.0
    l2 = 2.0 * mu2 / 3.0 + 4.0 * mu4 / 7.0 + 10.0 * mu6 / 21.0 + 40.0 * mu8 / 99.0
    l4 = 8.0 * mu4 / 35.0 + 24.0 * mu6 / 77.0 + 48.0 * mu8 / 143.0
    return l0, l2, l4


def _evaluate_matrix_stack(x: jnp.ndarray, matrix: np.ndarray, kdisc: jnp.ndarray, damping: jnp.ndarray) -> jnp.ndarray:
    values = jnp.real(kdisc**3 * _quadratic_form_columns(x, jnp.asarray(matrix)))
    return values * damping


def _build_m22_mu_matrices(m22: np.ndarray, etam: np.ndarray, growth_rate: float) -> dict[str, np.ndarray]:
    nu1 = -0.5 * etam[:, None]
    nu2 = -0.5 * etam[None, :]
    nu12 = nu1 + nu2
    denom = 98.0 * nu1 * nu2 * nu12 * nu12 - 91.0 * nu12 * nu12 + 36.0 * nu1 * nu2 - 14.0 * nu1 * nu2 * nu12 + 3.0 * nu12 + 58.0
    prefactor = m22 * 196.0 / denom
    f = growth_rate
    return {
        "mu2_vd": prefactor * (-f) * (7.0 * f * (-1.0 + 2.0 * nu1) * (-1.0 + 2.0 * nu2) * (6.0 + 7.0 * nu12) - 4.0 * (46.0 + 13.0 * nu2 + 98.0 * nu1**3 * nu2 - 63.0 * nu2**2 + 7.0 * nu1**2 * (-9.0 - 10.0 * nu2 + 28.0 * nu2**2) + nu1 * (13.0 - 138.0 * nu2 - 70.0 * nu2**2 + 98.0 * nu2**3))) / 392.0,
        "mu2_dd": prefactor * f * (7.0 * f * (2.0 + 2.0 * nu1**3 - nu2 - nu2**2 + 2.0 * nu2**3 - nu1**2 * (1.0 + 2.0 * nu2) - nu1 * (1.0 + 2.0 * nu2 + 2.0 * nu2**2)) + 4.0 * (10.0 - nu2 + 14.0 * nu1**3 * nu2 - 17.0 * nu2**2 + nu1**2 * (-17.0 + 6.0 * nu2 + 28.0 * nu2**2) + nu1 * (-1.0 - 22.0 * nu2 + 6.0 * nu2**2 + 14.0 * nu2**3))) / 56.0,
        "mu4_vv": prefactor * f * f * (147.0 * f * f * (-1.0 + 2.0 * nu1) * (-1.0 + 2.0 * nu2) - 28.0 * f * (-1.0 + 2.0 * nu1) * (-1.0 + 2.0 * nu2) * (-2.0 + 7.0 * nu12) + 8.0 * (50.0 - 9.0 * nu2 + 98.0 * nu1**3 * nu2 - 35.0 * nu2**2 + 7.0 * nu1**2 * (-5.0 - 18.0 * nu2 + 28.0 * nu2**2) + nu1 * (-9.0 - 66.0 * nu2 - 126.0 * nu2**2 + 98.0 * nu2**3))) / 1568.0,
        "mu4_vd": prefactor * f * f * (58.0 + 21.0 * nu2 + 112.0 * nu1**3 * nu2 - 106.0 * nu2**2 + 2.0 * nu1**2 * (-53.0 - 6.0 * nu2 + 112.0 * nu2**2) + 7.0 * f * (2.0 + nu1 + 4.0 * nu1**3 + nu2 - 8.0 * nu1 * nu2 - 8.0 * nu1**2 * nu2 - 8.0 * nu1 * nu2**2 + 4.0 * nu2**3) + nu1 * (21.0 - 204.0 * nu2 - 12.0 * nu2**2 + 112.0 * nu2**3)) / 56.0,
        "mu4_dd": prefactor * f * f * (2.0 * nu1 - 1.0) * (2.0 * nu2 - 1.0) * (2.0 + nu1**2 + 3.0 * nu2 + nu2**2 + nu1 * (3.0 + 2.0 * nu2)) / 8.0,
        "mu6_vv": prefactor * f**3 * (7.0 * f * (1.0 + 4.0 * nu1**3 + nu1**2 * (2.0 - 12.0 * nu2) + 2.0 * nu2 + 2.0 * nu2**2 + 4.0 * nu2**3 - 2.0 * nu1 * (-1.0 + 4.0 * nu2 + 6.0 * nu2**2)) + 2.0 * (26.0 + 9.0 * nu2 + 56.0 * nu1**3 * nu2 - 38.0 * nu2**2 + 2.0 * nu1**2 * (-19.0 - 18.0 * nu2 + 56.0 * nu2**2) + nu1 * (9.0 - 84.0 * nu2 - 36.0 * nu2**2 + 56.0 * nu2**3))) / 112.0,
        "mu6_vd": prefactor * f**3 * (2.0 * nu1 - 1.0) * (2.0 * nu2 - 1.0) * (2.0 + 2.0 * nu1**2 + 5.0 * nu2 + 2.0 * nu2**2 + nu1 * (5.0 + 4.0 * nu2)) / 8.0,
        "mu8": prefactor * f**4 * (2.0 * nu1 - 1.0) * (2.0 * nu2 - 1.0) * (3.0 + 4.0 * nu1**2 + 8.0 * nu2 + 4.0 * nu2**2 + 8.0 * nu1 * (1.0 + nu2)) / 32.0,
    }


def _build_bias_matrices(m22basic: np.ndarray, etam2: np.ndarray, growth_rate: float) -> dict[str, np.ndarray]:
    nu1 = -0.5 * etam2[:, None]
    nu2 = -0.5 * etam2[None, :]
    f = growth_rate
    return {
        "l0_b1b2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * (-12.0 + 7.0 * (3.0 + f) * nu1 + 7.0 * (3.0 + f) * nu2) / (42.0 * nu1 * nu2),
        "l0_b2": m22basic * (7.0 * f * f * (12.0 + 6.0 * nu1**2 - 17.0 * nu2 + 6.0 * nu2**2 + nu1 * (-17.0 + 12.0 * nu2)) + 5.0 * f * (24.0 + 14.0 * nu1**2 - 37.0 * nu2 + 14.0 * nu2**2 + nu1 * (-37.0 + 28.0 * nu2))) / (210.0 * nu1 * nu2),
        "l0_b1bG2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * (-1.0 + 2.0 * nu1 + 2.0 * nu2) * (7.0 * f * (2.0 + nu1 + nu2) + 3.0 * (6.0 + 7.0 * nu1 + 7.0 * nu2)) / (42.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2)),
        "l0_bG2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * (-1.0 + 2.0 * nu1 + 2.0 * nu2) * (-10.0 * f + 7.0 * f * (5.0 * (nu1 + nu2) + f * (-2.0 + 3.0 * nu1 + 3.0 * nu2))) / (210.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2)),
        "l2_b1b2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * f * (nu1 + nu2) / (3.0 * nu1 * nu2),
        "l2_b2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * f * (-16.0 + 14.0 * (nu1 + nu2) + f * (-13.0 + 12.0 * (nu1 + nu2))) / (42.0 * nu1 * nu2),
        "l2_b1bG2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * (-1.0 + 2.0 * nu1 + 2.0 * nu2) * f * (2.0 + nu1 + nu2) / (3.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2)),
        "l2_bG2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * (-1.0 + 2.0 * nu1 + 2.0 * nu2) * f * (-2.0 - f + 7.0 * (nu1 + nu2) + 6.0 * f * (nu1 + nu2)) / (21.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2)),
        "l4_b2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * (-1.0 + 2.0 * nu1 + 2.0 * nu2) * 2.0 * f * f / (35.0 * nu1 * nu2),
        "l4_bG2": m22basic * (-3.0 + 2.0 * nu1 + 2.0 * nu2) * (-1.0 + 2.0 * nu1 + 2.0 * nu2) * 4.0 * f * f * (1.0 + nu1 + nu2) / (35.0 * nu1 * (1.0 + nu1) * nu2 * (1.0 + nu2)),
    }


def compute_native_rsd_terms(
    linear_input: LinearPowerInput,
    settings: PTSettings,
    output_k: jnp.ndarray | None = None,
    fftlog_input: NativeFFTLogInput | None = None,
) -> dict[str, jnp.ndarray]:
    if settings.native_kernel_source != "analytic":
        raise NotImplementedError("Native one-loop RSD kernels only support native_kernel_source='analytic'.")

    nmax = settings.fftlog_n
    kernels = _analytic_realspace_kernel_registry(
        nmax,
        settings.fftlog_k0_over_h,
        settings.fftlog_kmax_over_h,
        settings.fftlog_bias_matter + settings.fftlog_bias,
        settings.fftlog_bias_bias + settings.fftlog_bias,
    )
    transfer = _analytic_rsd_transfer_registry(nmax, settings.fftlog_k0_over_h, settings.fftlog_kmax_over_h)

    if fftlog_input is None:
        fftlog_input = prepare_native_fftlog_input(linear_input, settings)

    kdisc = jnp.asarray(np.asarray(fftlog_input.kdisc, dtype=float))
    if output_k is None:
        output_k = jnp.asarray(np.asarray(linear_input.k, dtype=float))
    pk_linear = jnp.asarray(np.asarray(fftlog_input.pdisc, dtype=float))
    transfer_linear = jnp.asarray(np.asarray(fftlog_input.tdisc, dtype=float))
    h = float(fftlog_input.h)
    f = float(fftlog_input.growth_rate)
    logk = jnp.log(kdisc)

    etam = jnp.asarray(kernels["etam"])
    etam2 = jnp.asarray(kernels["etam2"])
    etam_transfer = jnp.asarray(transfer["etam_transfer"])
    etam2_transfer = jnp.asarray(transfer["etam2_transfer"])

    cmsym = _fftlog_coefficients_jax(kdisc, pk_linear, kdisc, etam, settings.fftlog_bias_matter + settings.fftlog_bias)
    cmsym2 = _fftlog_coefficients_jax(kdisc, pk_linear, kdisc, etam2, settings.fftlog_bias_bias + settings.fftlog_bias)
    cmsym_transfer = _fftlog_coefficients_jax(kdisc, transfer_linear, kdisc, etam_transfer, _TRANSFER_BIAS)
    cmsym2_transfer = _fftlog_coefficients_jax(kdisc, transfer_linear, kdisc, etam2_transfer, _TRANSFER_BIAS_BIASED)

    x = cmsym[:, None] * jnp.exp(etam[:, None] * logk[None, :])
    x2 = cmsym2[:, None] * jnp.exp(etam2[:, None] * logk[None, :])
    x_transfer = cmsym_transfer[:, None] * jnp.exp(etam_transfer[:, None] * logk[None, :])
    x2_transfer = cmsym2_transfer[:, None] * jnp.exp(etam2_transfer[:, None] * logk[None, :])

    pbin = pk_linear
    tbin = transfer_linear
    sigmav = jnp.trapezoid(kdisc * pk_linear, jnp.log(kdisc)) / (6.0 * jnp.pi**2)
    damping = jnp.exp(-jnp.power(kdisc / (3.0 * h), 6.0))

    nu = -0.5 * etam
    m13 = jnp.asarray(kernels["m13"])
    m13_mu2_dd = m13 * 2.0 / (1.0 + 9.0 * nu) * 9.0 * f * (1.0 + nu)
    m13_mu2_vd = m13 * 2.0 / (1.0 + 9.0 * nu) * (-f * (7.0 + 9.0 * f - 9.0 * nu))
    m13_mu4_vv = m13 / (1.0 + 9.0 * nu) * (-3.0 * f * f * (5.0 + 6.0 * f - 3.0 * nu))
    m13_mu4_vd = m13 * 2.0 / (1.0 + 9.0 * nu) * 9.0 * f * f * (1.0 + 2.0 * nu)
    m13_mu6 = m13 * 2.0 / (1.0 + 9.0 * nu) * 9.0 * f**3 * nu

    p13_mu0_dd = (jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x, m13) * pbin) - 61.0 * pbin * kdisc**2 * sigmav / 105.0) * damping
    p13_mu2_dd = (jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x, m13_mu2_dd) * pbin) - pbin * kdisc**2 * sigmav * f * (105.0 * f - 6.0) / 105.0) * damping
    p13_mu2_vd = (jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x, m13_mu2_vd) * pbin) - pbin * kdisc**2 * sigmav * f * (250.0 + 144.0 * f) / 105.0) * damping
    p13_mu4_vv = (jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x, m13_mu4_vv) * pbin) - pbin * kdisc**2 * sigmav * f * f * (63.0 + 48.0 * f) / 35.0) * damping
    p13_mu4_vd = (jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x, m13_mu4_vd) * pbin) - pbin * kdisc**2 * sigmav * f * f * (44.0 + 70.0 * f) / 35.0) * damping
    p13_mu6 = (jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x, m13_mu6) * pbin) - pbin * kdisc**2 * sigmav * f**3 * (46.0 + 35.0 * f) / 35.0) * damping

    m22_mu = _build_m22_mu_matrices(np.asarray(kernels["m22"]), np.asarray(kernels["etam"]), f)
    p22_mu0_dd = _evaluate_matrix_stack(x, np.asarray(kernels["m22"]), kdisc, damping)
    p22_mu2_vd = _evaluate_matrix_stack(x, m22_mu["mu2_vd"], kdisc, damping)
    p22_mu2_dd = _evaluate_matrix_stack(x, m22_mu["mu2_dd"], kdisc, damping)
    p22_mu4_vv = _evaluate_matrix_stack(x, m22_mu["mu4_vv"], kdisc, damping)
    p22_mu4_vd = _evaluate_matrix_stack(x, m22_mu["mu4_vd"], kdisc, damping)
    p22_mu4_dd = _evaluate_matrix_stack(x, m22_mu["mu4_dd"], kdisc, damping)
    p22_mu6_vv = _evaluate_matrix_stack(x, m22_mu["mu6_vv"], kdisc, damping)
    p22_mu6_vd = _evaluate_matrix_stack(x, m22_mu["mu6_vd"], kdisc, damping)
    p22_mu8 = _evaluate_matrix_stack(x, m22_mu["mu8"], kdisc, damping)

    p12_l0_vv = tbin * _evaluate_matrix_stack(x_transfer, f * f * transfer["vv0_f2"] + f**3 * transfer["vv0_f3"], kdisc, damping)
    p12_l0_vd = tbin * _evaluate_matrix_stack(x_transfer, f * transfer["vd0_f1"] + f * f * transfer["vd0_f2"], kdisc, damping)
    p12_l0_dd = tbin * _evaluate_matrix_stack(x_transfer, transfer["dd0_f0"] + f * transfer["dd0_f1"], kdisc, damping)
    p12_l2_vv = tbin * _evaluate_matrix_stack(x_transfer, (20.0 / 7.0) * f * f * transfer["vv0_f2"] + f**3 * transfer["vv2_f3"], kdisc, damping)
    p12_l2_vd = tbin * _evaluate_matrix_stack(x_transfer, 2.0 * f * transfer["vd0_f1"] + f * f * transfer["vd2_f2"], kdisc, damping)
    p12_l2_dd = tbin * _evaluate_matrix_stack(x_transfer, 2.0 * f * transfer["dd0_f1"], kdisc, damping)
    p12_l4_vv = tbin * _evaluate_matrix_stack(x_transfer, (8.0 / 7.0) * f * f * transfer["vv0_f2"] + f**3 * transfer["vv4_f3"], kdisc, damping)
    p12_l4_vd = tbin * _evaluate_matrix_stack(x_transfer, f * f * transfer["vd4_f2"], kdisc, damping)

    l0_vv_mu, l2_vv_mu, l4_vv_mu = _mu_to_multipoles(jnp.zeros_like(kdisc), jnp.zeros_like(kdisc), p13_mu4_vv + p22_mu4_vv, p13_mu6 + p22_mu6_vv, p22_mu8)
    l0_vd_mu, l2_vd_mu, l4_vd_mu = _mu_to_multipoles(jnp.zeros_like(kdisc), p13_mu2_vd + p22_mu2_vd, p13_mu4_vd + p22_mu4_vd, p22_mu6_vd, jnp.zeros_like(kdisc))
    l0_dd_mu, l2_dd_mu, l4_dd_mu = _mu_to_multipoles(p13_mu0_dd + p22_mu0_dd, p13_mu2_dd + p22_mu2_dd, p22_mu4_dd, jnp.zeros_like(kdisc), jnp.zeros_like(kdisc))

    bias_m22 = _build_bias_matrices(np.asarray(kernels["m22basic"]), np.asarray(kernels["etam2"]), f)
    bias_terms = {
        "rsd_l0_b1_b2": _evaluate_matrix_stack(x2, bias_m22["l0_b1b2"], kdisc, jnp.ones_like(damping)),
        "rsd_l0_b2": _evaluate_matrix_stack(x2, bias_m22["l0_b2"], kdisc, jnp.ones_like(damping)) + tbin * _evaluate_matrix_stack(x2_transfer, f * transfer["b2_vv0_f1"], kdisc, damping),
        "rsd_l0_b1_bG2": -_evaluate_matrix_stack(x2, bias_m22["l0_b1bG2"], kdisc, jnp.ones_like(damping)),
        "rsd_l0_bG2": -(_evaluate_matrix_stack(x2, bias_m22["l0_bG2"], kdisc, jnp.ones_like(damping)) + tbin * _evaluate_matrix_stack(x_transfer, f * transfer["bG2_vv0_f1"], kdisc, damping)),
        "rsd_l2_b1_b2": _evaluate_matrix_stack(x2, bias_m22["l2_b1b2"], kdisc, jnp.ones_like(damping)),
        "rsd_l2_b2": _evaluate_matrix_stack(x2, bias_m22["l2_b2"], kdisc, jnp.ones_like(damping)) + tbin * _evaluate_matrix_stack(x2_transfer, 2.0 * f * transfer["b2_vv0_f1"], kdisc, damping),
        "rsd_l2_b1_bG2": -_evaluate_matrix_stack(x2, bias_m22["l2_b1bG2"], kdisc, jnp.ones_like(damping)),
        "rsd_l2_bG2": -(_evaluate_matrix_stack(x2, bias_m22["l2_bG2"], kdisc, jnp.ones_like(damping)) + tbin * _evaluate_matrix_stack(x_transfer, 2.0 * f * transfer["bG2_vv0_f1"], kdisc, damping)),
        "rsd_l4_b2": _evaluate_matrix_stack(x2, bias_m22["l4_b2"], kdisc, jnp.ones_like(damping)),
        "rsd_l4_bG2": -_evaluate_matrix_stack(x2, bias_m22["l4_bG2"], kdisc, jnp.ones_like(damping)),
    }

    p_ifg2 = jnp.abs(jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x2, jnp.asarray(kernels["ifg2"])) * pbin))

    native = {
        "rsd_l0_loop_00": _interpolate_to_output_jax(kdisc, l0_vv_mu + p12_l0_vv, output_k),
        "rsd_l0_loop_01": _interpolate_to_output_jax(kdisc, l0_vd_mu + p12_l0_vd, output_k),
        "rsd_l0_loop_11": _interpolate_to_output_jax(kdisc, l0_dd_mu + p12_l0_dd, output_k),
        "rsd_l2_loop_00": _interpolate_to_output_jax(kdisc, l2_vv_mu + p12_l2_vv, output_k),
        "rsd_l2_loop_01": _interpolate_to_output_jax(kdisc, l2_vd_mu + p12_l2_vd, output_k),
        "rsd_l2_11": _interpolate_to_output_jax(kdisc, l2_dd_mu + p12_l2_dd, output_k),
        "rsd_l4_loop_00": _interpolate_to_output_jax(kdisc, l4_vv_mu + p12_l4_vv, output_k),
        "rsd_l4_loop_01": _interpolate_to_output_jax(kdisc, l4_vd_mu + p12_l4_vd, output_k),
        "rsd_l4_loop_11": _interpolate_to_output_jax(kdisc, l4_dd_mu, output_k),
        "rsd_l0_gamma3_b1": _interpolate_to_output_jax(kdisc, -p_ifg2, output_k),
        "rsd_l0_gamma3_bias": _interpolate_to_output_jax(kdisc, -(f / 3.0) * p_ifg2, output_k),
        "rsd_l2_gamma3": _interpolate_to_output_jax(kdisc, -(2.0 * f / 3.0) * p_ifg2, output_k),
    }
    native.update({name: _interpolate_to_output_jax(kdisc, values, output_k) for name, values in bias_terms.items()})
    return native
