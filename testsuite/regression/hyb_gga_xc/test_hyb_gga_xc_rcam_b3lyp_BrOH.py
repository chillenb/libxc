
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_rcam_b3lyp_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.723444790894804e+01, -1.723446404006196e+01, -1.723460466755083e+01, -1.723433914496088e+01, -1.723445584199617e+01, -1.723445584199617e+01, -2.750826294671286e+00, -2.750800968468664e+00, -2.750224196773942e+00, -2.751831035442588e+00, -2.750825580070702e+00, -2.750825580070702e+00, -4.304901986492492e-01, -4.302298854497371e-01, -4.238888822080969e-01, -4.279356686326509e-01, -4.303937439095498e-01, -4.303937439095498e-01, -3.952728480921330e-02, -4.156148789059073e-02, -5.328986548624494e-01, -3.387640402733024e-03, -4.014867092255335e-02, -4.014867092255335e-02, -6.337713428492863e-04, -6.563016718419037e-04, 6.915763149706313e-03, -3.843627524086900e-04, -6.499453509105975e-04, -6.499453509105975e-04, -4.035295905195224e+00, -4.034639525938072e+00, -4.035234404251673e+00, -4.034723941074613e+00, -4.034951317451341e+00, -4.034951317451341e+00, -1.607443813870839e+00, -1.615899834846003e+00, -1.608102354735547e+00, -1.614690349635186e+00, -1.612316132714632e+00, -1.612316132714632e+00, -3.432936069540621e-01, -3.702492988615421e-01, -3.202411009978614e-01, -3.296719162949047e-01, -3.593799745510153e-01, -3.593799745510153e-01, 1.604998390085420e-02, -3.163929719439781e-02, 1.418039123386780e-02, -1.369627093341623e+00, 6.612149715616297e-03, 6.612149715616297e-03, -3.736402805765515e-04, -4.083640882239981e-04, -3.110255400189073e-04, 1.308769963893017e-02, -3.727142349811358e-04, -3.727142349811359e-04, -3.423091651121534e-01, -3.429530946398570e-01, -3.427580681588807e-01, -3.425794759930757e-01, -3.426713953797932e-01, -3.426713953797933e-01, -3.272869269491999e-01, -2.873904417386815e-01, -3.004015394540673e-01, -3.114067569065569e-01, -3.058563396007196e-01, -3.058563396007196e-01, -3.943308654940033e-01, -6.837458968323774e-02, -1.046187693884021e-01, -1.649458233100480e-01, -1.336957326164420e-01, -1.336957326164420e-01, -2.474917810921401e-01, 6.063569353752236e-03, 1.286152298826725e-02, -1.487692933301543e-01, 1.345020493384683e-02, 1.345020493384687e-02, -7.242406520714354e-04, -1.804514524609889e-04, -2.603127099505823e-04, 1.152854168605679e-02, -3.247036153210321e-04, -3.247036153210317e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_rcam_b3lyp_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.033717863574769e+01, -2.033724795473077e+01, -2.033756513833426e+01, -2.033642986841775e+01, -2.033721512587421e+01, -2.033721512587421e+01, -3.204944961285038e+00, -3.204970977487187e+00, -3.205821591282460e+00, -3.205052071000597e+00, -3.204972322362586e+00, -3.204972322362586e+00, -5.466552122814969e-01, -5.460910308383945e-01, -5.330185827300362e-01, -5.374946651906338e-01, -5.464486858661554e-01, -5.464486858661554e-01, -1.014288982691741e-01, -1.025240422041202e-01, -6.706570786038878e-01, -7.107871701712308e-02, -1.017359480904229e-01, -1.017359480904229e-01, -6.715835998638983e-04, -7.000335717414045e-04, -4.560395554731794e-03, -3.500742152032487e-04, -6.932730323839984e-04, -6.932730323839984e-04, -4.942995294819639e+00, -4.945034277295544e+00, -4.943203946456114e+00, -4.944789350204296e+00, -4.944040270721979e+00, -4.944040270721979e+00, -1.722837151301480e+00, -1.736163273808728e+00, -1.716663918045260e+00, -1.727012462280735e+00, -1.740022038180073e+00, -1.740022038180073e+00, -4.636642362111824e-01, -5.174646594959348e-01, -4.350857547488559e-01, -4.604599508997697e-01, -4.854599352104101e-01, -4.854599352104101e-01, -4.979386000721790e-02, -1.193464082888291e-01, -4.662731094368265e-02, -1.793432054894051e+00, -5.691036611067422e-02, -5.691036611067422e-02, -3.387441275916753e-04, -3.815510314833608e-04, -2.905331698499928e-04, -1.659567366960130e-02, -3.492998167679705e-04, -3.492998167679705e-04, -4.889338646717912e-01, -4.833395557536938e-01, -4.852482191453902e-01, -4.867914690509452e-01, -4.860112800859320e-01, -4.860112800859320e-01, -4.700645215647675e-01, -3.908300221065640e-01, -4.106407177801942e-01, -4.311725847786517e-01, -4.204319633485444e-01, -4.204319633485444e-01, -5.474596201612500e-01, -1.579649963507876e-01, -1.892571505670578e-01, -2.475945119449353e-01, -2.151240408524253e-01, -2.151240408524253e-01, -3.445144224071615e-01, -2.959624687950802e-03, -1.250749366322841e-02, -2.257201714772463e-01, -3.027976638827623e-02, -3.027976638827614e-02, -7.753898208069444e-04, -1.196064726937732e-04, -2.115432288831614e-04, -2.590524391613480e-02, -3.003439826936939e-04, -3.003439826936936e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_rcam_b3lyp_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.799199186468669e-09, -6.799170889986837e-09, -6.798942132887029e-09, -6.799407559966173e-09, -6.799185120840562e-09, -6.799185120840562e-09, -8.150325805420520e-06, -8.150602954476066e-06, -8.156922840980822e-06, -8.139344668120780e-06, -8.150333670387078e-06, -8.150333670387078e-06, -2.573268345782286e-03, -2.567447615710550e-03, -2.379933296002146e-03, -2.347991470438729e-03, -2.571213560281148e-03, -2.571213560281148e-03, 2.170558586834741e-01, 2.014880790044504e-01, -1.673329834780048e-03, 8.751574645529174e-01, 2.122565680111616e-01, 2.122565680111616e-01, -7.247017807229941e+00, -6.352781915492212e+00, 9.994372607925451e+00, -5.371970091971082e+01, -6.618558127083320e+00, -6.618558127083320e+00, -1.943126821877973e-06, -1.944547927268959e-06, -1.943261055023607e-06, -1.944366158767585e-06, -1.943870152320401e-06, -1.943870152320401e-06, -5.510478682232915e-05, -5.415180788513945e-05, -5.500443239702791e-05, -5.426264307762811e-05, -5.458709924386919e-05, -5.458709924386919e-05, -5.194608376103990e-03, -5.446159493706144e-03, -5.894971713127672e-03, -6.752113192124451e-03, -4.853507610183049e-03, -4.853507610183049e-03, 2.426424849674061e+00, 2.306608730523233e-01, 2.516840375217236e+00, -1.052315187432150e-04, 1.542353652415568e+00, 1.542353652415568e+00, -6.056344440548546e+01, -4.130521470917112e+01, -1.271784825255522e+02, 7.237715227465609e+00, -6.023248286867747e+01, -6.023248286867749e+01, -7.186089573821040e-03, -6.610806972535672e-03, -6.788670011696514e-03, -6.944864061498308e-03, -6.864412821974314e-03, -6.864412821974314e-03, -8.119362698423555e-03, -6.120036674155593e-03, -6.567066816056418e-03, -6.992865954494009e-03, -6.783218750435182e-03, -6.783218750435183e-03, -4.628768219812357e-03, 6.978902458962566e-02, 1.787959165080055e-02, -8.668044675322523e-03, -3.601925795910059e-04, -3.601925795909608e-04, -7.390188007699562e-03, 1.006104118527515e+01, 8.413646882500494e+00, -1.185960247775023e-02, 4.229767380556360e+00, 4.229767380556360e+00, -4.221799154601847e+00, -2.006520551927640e+03, -3.022000140672307e+02, 4.719374129908829e+00, -1.069187639298742e+02, -1.069187639298746e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05