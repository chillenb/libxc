
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbeef_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.247636947774660e+01, -2.247639782073433e+01, -2.247660157521651e+01, -2.247613586213504e+01, -2.247638377938630e+01, -2.247638377938630e+01, -3.479820286675384e+00, -3.479839131177608e+00, -3.480660479507336e+00, -3.481936732441213e+00, -3.479833763496481e+00, -3.479833763496481e+00, -6.602260893925203e-01, -6.600005551474402e-01, -6.572919098548893e-01, -6.664990515171195e-01, -6.601479274914556e-01, -6.601479274914556e-01, -2.100455542498866e-01, -2.108443250449828e-01, -7.391346398023541e-01, -1.552411014647888e-01, -2.100861949801708e-01, -2.100861949801708e-01, -1.094209735851050e-02, -1.147306703132323e-02, -4.934052645460027e-02, -5.205046619454962e-03, -1.135291609221389e-02, -1.135291609221389e-02, -5.409717902652875e+00, -5.409321292629242e+00, -5.409683378203858e+00, -5.409374867594561e+00, -5.409505345534201e+00, -5.409505345534201e+00, -2.221414811533236e+00, -2.243462241191067e+00, -2.227352954197541e+00, -2.244728660452260e+00, -2.229320362414036e+00, -2.229320362414036e+00, -6.078965415439048e-01, -6.505472209002350e-01, -5.591904364338492e-01, -5.769900258123500e-01, -6.333419103493554e-01, -6.333419103493554e-01, -1.099444455222112e-01, -2.126413496432615e-01, -1.087404709131593e-01, -1.910223555005637e+00, -1.284206941463707e-01, -1.284206941463707e-01, -5.020795535495583e-03, -5.741556732259043e-03, -4.302756407816107e-03, -6.876223613687389e-02, -5.231573644840989e-03, -5.231573644840986e-03, -6.064885488043661e-01, -6.072738742209264e-01, -6.069474088832265e-01, -6.067263237453072e-01, -6.068328531951226e-01, -6.068328531951226e-01, -5.874894181025639e-01, -5.533121041822496e-01, -5.608633830220162e-01, -5.690263669907909e-01, -5.644816987064823e-01, -5.644816987064823e-01, -6.758590148192580e-01, -2.773669786762044e-01, -3.134190295695026e-01, -3.664392543409464e-01, -3.412994190536188e-01, -3.412994190536188e-01, -4.894481521418212e-01, -4.535718265869011e-02, -6.202862256984627e-02, -3.414798150262229e-01, -8.879726858618001e-02, -8.879726858618002e-02, -1.285032407589327e-02, -1.568127534928187e-03, -2.980736587607246e-03, -8.475543314949195e-02, -4.444622880577821e-03, -4.444622880577815e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbeef_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.728662961507708e+01, -2.728673723365800e+01, -2.728717752718204e+01, -2.728541382958305e+01, -2.728668673041397e+01, -2.728668673041397e+01, -4.407668779110713e+00, -4.407984440352155e+00, -4.417920344444481e+00, -4.417154276085471e+00, -4.407759790225430e+00, -4.407759790225430e+00, -8.086649277195551e-01, -8.065270929047504e-01, -7.540643639010451e-01, -7.739131654746706e-01, -8.079065166496645e-01, -8.079065166496645e-01, -1.601223531832147e-01, -1.699911182391384e-01, -8.875592588519343e-01, -2.148967475231556e-01, -1.640982786563601e-01, -1.640982786563601e-01, -1.499686572288348e-02, -1.571274708385542e-02, -6.906370844251923e-02, -6.969690501094425e-03, -1.558796001427499e-02, -1.558796001427500e-02, -6.746663684057320e+00, -6.750098660848294e+00, -6.747001965406978e+00, -6.749673574406228e+00, -6.748443520950355e+00, -6.748443520950355e+00, -2.045099241597691e+00, -2.090709207568322e+00, -2.027160730102715e+00, -2.070817040644623e+00, -2.100791940698660e+00, -2.100791940698660e+00, -7.736801554399761e-01, -8.748551278546752e-01, -7.380343393095172e-01, -8.313783763825594e-01, -7.976352230120334e-01, -7.976352230120334e-01, -1.582362018728824e-01, -2.226183693370352e-01, -1.583814525715043e-01, -2.810077388030110e+00, -1.992740593227888e-01, -1.992740593227888e-01, -6.721808647831941e-03, -7.696176241206964e-03, -5.766076907016824e-03, -8.803111986807477e-02, -7.012903547465065e-03, -7.012903547465064e-03, -8.031074958391137e-01, -7.936974836698513e-01, -7.972824145604638e-01, -7.998956783888121e-01, -7.986065921919746e-01, -7.986065921919746e-01, -7.765590755842462e-01, -6.042405729909135e-01, -6.581692133154774e-01, -7.102199356482306e-01, -6.846775207007827e-01, -6.846775207007828e-01, -9.521336893987689e-01, -2.014173682355562e-01, -2.454915879756713e-01, -4.282895594100103e-01, -3.294085379901566e-01, -3.294085379901567e-01, -5.488532228351899e-01, -6.673164594655086e-02, -8.308287895281434e-02, -4.272147453214462e-01, -1.235476259825905e-01, -1.235476259825905e-01, -1.754105204525251e-02, -2.091953990822022e-03, -3.981389446130332e-03, -1.176235253436497e-01, -5.954033556005432e-03, -5.954033556005427e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeef_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.700617426291422e-09, -6.700516429034880e-09, -6.700191622328070e-09, -6.701852157914014e-09, -6.700562993885486e-09, -6.700562993885486e-09, -9.099833257968574e-06, -9.109255076792737e-06, -9.406097379691299e-06, -9.443578185285169e-06, -9.101294416866937e-06, -9.101294416866937e-06, -7.444322337664077e-03, -7.441945464692945e-03, -7.281336260278442e-03, -7.367247129172909e-03, -7.443954930426428e-03, -7.443954930426428e-03, -2.634129843195455e+00, -2.404516752156768e+00, -3.034355419422005e-03, 3.449768746253631e-01, -2.536300474803730e+00, -2.536300474803730e+00, 2.717660872507858e+01, 2.172710215727957e+01, 2.158339656743713e+00, 1.003754502406175e+01, 2.718866229691915e+01, 2.718866229692003e+01, -1.882733035037384e-06, -1.879623213598523e-06, -1.882542473552692e-06, -1.880118158499960e-06, -1.880967891170202e-06, -1.880967891170202e-06, -2.187463471913628e-04, -1.970570383682573e-04, -2.160539425524536e-04, -1.966019715237384e-04, -2.069069577154914e-04, -2.069069577154914e-04, -2.006853790091358e-02, -7.155543567912581e-03, -2.900271369361896e-02, -2.710117383326627e-02, -1.154961340960529e-02, -1.154961340960529e-02, 4.975664149851809e-01, -8.408070616455390e-01, 6.529478936265206e-01, -2.170875056764094e-04, 1.554781909288477e+00, 1.554781909288477e+00, 1.065487005016902e+01, 1.062957934339238e+01, 3.045298579672249e+01, -4.992237823967404e-01, 1.569773876162341e+01, 1.569773876162524e+01, -5.427666792203390e-03, -7.043168365902196e-03, -6.480890830162670e-03, -6.033593158351526e-03, -6.257220632928915e-03, -6.257220632928916e-03, -1.977061005860396e-03, -2.687559111572096e-02, -2.012356860981272e-02, -1.276407015078537e-02, -1.642944983564857e-02, -1.642944983564858e-02, -1.249110112551407e-02, -9.678748723153026e-01, -6.086779212771204e-01, -1.680441998672827e-01, -3.598018020819853e-01, -3.598018020819854e-01, -6.881084725134420e-02, 5.709530519223886e+00, 5.696446488570999e-01, -2.252991951413820e-01, 7.362260207432295e-01, 7.362260207432214e-01, 9.803306958313446e+00, 1.859410741985398e+01, 1.593925045843888e+01, 7.654632932313566e-01, 2.298107648277988e+01, 2.298107648278301e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeef_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeef_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.081827756048792e-05, -2.082244080956901e-05, -2.082371212721407e-05, -2.075504754919888e-05, -2.082062858812967e-05, -2.082062858812967e-05, 2.533155699738840e-03, 2.539690356660019e-03, 2.746564641547258e-03, 2.773598103520773e-03, 2.534280377161188e-03, 2.534280377161188e-03, 1.376312017206343e-02, 1.366785793918608e-02, 1.134452974743936e-02, 1.340772411177078e-02, 1.373146890223644e-02, 1.373146890223644e-02, 2.305703856961982e-01, 2.133984755121453e-01, 2.572343321535749e-03, -8.963367399724750e-03, 2.225040123609409e-01, 2.225040123609409e-01, -2.362563627964284e-04, -1.671000205376797e-04, -1.531250335452280e-03, 6.532209754300902e-12, -2.552566800323187e-04, -2.552566800323287e-04, -2.199105075812639e-05, -2.206389710743813e-05, -2.171783532190876e-05, -2.178842171581756e-05, -2.240987238820245e-05, -2.240987238820245e-05, 1.876857098265019e-02, 1.630410092346009e-02, 1.823722068474848e-02, 1.603121302363011e-02, 1.774116305868750e-02, 1.774116305868750e-02, 4.258096830372330e-02, 1.754449819135496e-02, 5.688817345469790e-02, 6.654005720339246e-02, 1.731157562751259e-02, 1.731157562751259e-02, -2.346738708161516e-03, 8.419699225432851e-02, -3.068257922027189e-03, 1.981960155497095e-02, -2.675583401926917e-02, -2.675583401926917e-02, 1.001248682593190e-10, -8.063567121918848e-11, -9.678583771486241e-10, -2.668077583241952e-04, -3.596698506495397e-11, -3.596698502084262e-11, -1.055688565075433e-02, -8.397229763005765e-03, -9.162459733035108e-03, -9.760271375918186e-03, -9.465004096972858e-03, -9.465004096972856e-03, -2.170611671087762e-02, 1.177733875798062e-02, 3.852855875866234e-03, -5.983950685285997e-03, -9.275191764374276e-04, -9.275191764374277e-04, 4.784896396979349e-02, 1.863268316074844e-01, 1.746029897598701e-01, 8.289684228848039e-02, 1.375061682491720e-01, 1.375061682491720e-01, 7.208139772936102e-02, -4.342943048196039e-03, -1.701492166152402e-03, 9.609094015261914e-02, -3.641783520329098e-03, -3.641783520329035e-03, -5.146859095122276e-08, 1.782932426670933e-13, 8.747935326300653e-10, -2.876607947476804e-03, -4.261340706665207e-12, -4.261340705035694e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05