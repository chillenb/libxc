
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc18_2_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.347136487037443e-05, -1.347125214916682e-05, -1.347071624531621e-05, -1.347239407424561e-05, -1.347151983744951e-05, -1.347151983744951e-05, -3.199298567284035e-03, -3.199277345836998e-03, -3.198546037676677e-03, -3.197950936835149e-03, -3.199036126268948e-03, -3.199036126268948e-03, -4.912583992884159e-02, -4.893644873251890e-02, -4.453208341396703e-02, -4.584090258526676e-02, -4.556029107731481e-02, -4.556029107731481e-02, -6.641898759852700e-04, -6.949952122229434e-04, -8.278183819704878e-02, -2.897615029191491e-04, -4.106803094624952e-04, -4.106803094624951e-04, -2.787114416236374e-08, -3.249917928642086e-08, -5.457888069207577e-06, -5.368849191209628e-09, -1.065241472461349e-08, -1.065241472461349e-08, -9.369196064755986e-04, -9.363999666595527e-04, -9.368937165093072e-04, -9.364349972036278e-04, -9.366572087904933e-04, -9.366572087904933e-04, -1.993896036160052e-02, -1.946198774427065e-02, -2.029733703558446e-02, -1.986462163462658e-02, -1.950627427168332e-02, -1.950627427168332e-02, -3.079835619275625e-02, -4.108779626397507e-02, -2.360743682711576e-02, -2.694202397909042e-02, -3.255094226483609e-02, -3.255094226483609e-02, -9.682457608973818e-05, -6.672364665211163e-04, -7.665102642227575e-05, -2.127570252373845e-02, -1.565593101694870e-04, -1.565593101694870e-04, -2.466047097595370e-09, -5.015906327433700e-09, -2.242508994260565e-09, -2.331139542156729e-05, -3.920668017602083e-09, -3.920668017602083e-09, -3.131834052736786e-02, -3.070179361177253e-02, -3.091767069943861e-02, -3.109635779742502e-02, -3.100681779354815e-02, -3.100681779354815e-02, -2.848687153503039e-02, -1.701342971781625e-02, -1.983832658881969e-02, -2.298556097531524e-02, -2.133334645409047e-02, -2.133334645409047e-02, -4.848134850748914e-02, -1.352912738543991e-03, -2.361578155090839e-03, -5.571624765365984e-03, -3.612914667488695e-03, -3.612914667488694e-03, -1.296660400551917e-02, -4.785539793216012e-06, -1.212741208908214e-05, -4.970209166792704e-03, -4.813544620301920e-05, -4.813544620301921e-05, -7.834648596553736e-08, -9.580264570140950e-11, -8.860174677674878e-10, -3.985587699385296e-05, -3.106173995125981e-09, -3.106173995125972e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc18_2_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.451528502489682e-09, 1.451504213807937e-09, 1.451388742319337e-09, 1.451750280333617e-09, 1.451561894462551e-09, 1.451561894462551e-09, 7.805647881815487e-05, 7.805546808409298e-05, 7.802064117381393e-05, 7.799230628166233e-05, 7.804397972714631e-05, 7.804397972714631e-05, -1.064620489375155e-01, -1.060564626911484e-01, -9.654601649100394e-02, -9.938686452899884e-02, -9.877875749402984e-02, -9.877875749402984e-02, -1.331874017648458e-03, -1.393814573442126e-03, -1.704306580872761e-01, -5.801917915994102e-04, -8.227016140902934e-04, -8.227016140902933e-04, -5.574229453913030e-08, -6.499836702241084e-08, -1.091601442727473e-05, -1.073769861301557e-08, -2.130483035701835e-08, -2.130483035701835e-08, 6.924739271890769e-06, 6.917113912537624e-06, 6.924359256295020e-06, 6.917627831666475e-06, 6.920888247490753e-06, 6.920888247490753e-06, 2.346385730180645e-03, 2.252514461002293e-03, 2.417612084842645e-03, 2.331684820391350e-03, 2.261184604743015e-03, 2.261184604743015e-03, -6.626591112252317e-02, -8.902034425296737e-02, -5.031459853148303e-02, -5.770313258252464e-02, -7.015733638504537e-02, -7.015733638504537e-02, -1.937240434985230e-04, -1.337999167325154e-03, -1.533490019604116e-04, 2.614909638675659e-03, -3.133142477033899e-04, -3.133142477033899e-04, -4.932094243841844e-09, -1.003181285614191e-08, -4.485018028751902e-09, -4.662713669495311e-05, -7.841336158177259e-09, -7.841336158177259e-09, -6.742069286999151e-02, -6.605145311112867e-02, -6.653089508314312e-02, -6.692772835268100e-02, -6.672887711746230e-02, -6.672887711746230e-02, -6.113234451579892e-02, -3.581601484364875e-02, -4.200314416150004e-02, -4.893975799892909e-02, -4.529335061449152e-02, -4.529335061449152e-02, -1.050806288943639e-01, -2.720174920617372e-03, -4.766223532981885e-03, -1.137172186732676e-02, -7.324759075814119e-03, -7.324759075814117e-03, -2.703963424495062e-02, -9.571262784431007e-06, -2.425600055748387e-05, -1.012387536909807e-02, -9.628941521834159e-05, -9.628941521834160e-05, -1.566930210363920e-07, -1.916052914762442e-10, -1.772034941815191e-09, -7.972445433244226e-05, -6.212348067438493e-09, -6.212348067438476e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05