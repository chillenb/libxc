
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_1d_loos_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_1d_loos", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [7.138806876123217e-03, 7.138806884800417e-03, 7.138806930454461e-03, 7.138806787005720e-03, 7.138806879615554e-03, 7.138806879615554e-03, 6.859592667549399e-03, 6.859594457400519e-03, 6.859678743749579e-03, 6.859714234151815e-03, 6.859597221091542e-03, 6.859597221091542e-03, -4.351021923208238e-02, -4.363187230145405e-02, -4.685597913101307e-02, -4.591451958265270e-02, -4.355456503193755e-02, -4.355456503193755e-02, -2.990157116021644e-02, -3.082572455714801e-02, -2.419825336106625e-02, -1.334586627763323e-02, -3.017795376427476e-02, -3.017795376427476e-02, -6.991759588646484e-06, -8.034747424832000e-06, -5.349222258115328e-04, -7.722186865842156e-07, -7.786242906277052e-06, -7.786242906277052e-06, 7.057173354821694e-03, 7.057217719353215e-03, 7.057178037139230e-03, 7.057212531155722e-03, 7.057195907782611e-03, 7.057195907782611e-03, 5.467771568383031e-03, 5.504795418635304e-03, 5.452481955996497e-03, 5.481832797825143e-03, 5.512077075291061e-03, 5.512077075291061e-03, -6.017572881962509e-02, -5.021513103700711e-02, -6.619819196520134e-02, -6.138879919116048e-02, -5.567032154338879e-02, -5.567032154338879e-02, -5.691459329625208e-03, -3.039785923542710e-02, -5.486431283659154e-03, 5.361089227138713e-03, -8.560170137615038e-03, -8.560170137615038e-03, -6.932713945249043e-07, -1.034762819259726e-06, -4.357056319203699e-07, -1.433375783668318e-03, -7.827596281886468e-07, -7.827596281886468e-07, -5.643254253813926e-02, -5.707595225134693e-02, -5.684509086849382e-02, -5.666714287296843e-02, -5.675618826093188e-02, -5.675618826093188e-02, -6.032241977952802e-02, -7.539747409624571e-02, -7.123330027381265e-02, -6.714250456762284e-02, -6.924711583237946e-02, -6.924711583237947e-02, -4.451539564934937e-02, -4.919100988054975e-02, -6.535254974542555e-02, -8.357904403441017e-02, -7.591696650344418e-02, -7.591696650344418e-02, -8.287774948191909e-02, -4.256091272027045e-04, -1.047671297221217e-03, -8.052409574215384e-02, -3.122899910191876e-03, -3.122899910191876e-03, -1.125356065285727e-05, -2.123863829954240e-08, -1.455861311826005e-07, -2.723028339539018e-03, -4.805046406005702e-07, -4.805046406005682e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_1d_loos_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_1d_loos", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [7.140000074779554e-03, 7.140000074300967e-03, 7.140000076181518e-03, 7.140000073243920e-03, 7.140000072566454e-03, 7.140000072566454e-03, 7.143873227051520e-03, 7.143873179046787e-03, 7.143870918432872e-03, 7.143869966746672e-03, 7.143873104919426e-03, 7.143873104919426e-03, -4.040102288550206e-03, -4.134192714732232e-03, -6.769010797524425e-03, -5.971180436252345e-03, -4.074356567646010e-03, -4.074356567646010e-03, -5.293774434168181e-02, -5.436762670613201e-02, 6.367246987605820e-03, -2.521228397365474e-02, -5.336673936918635e-02, -5.336673936918635e-02, -1.398147169174470e-05, -1.606692835362198e-05, -1.066089972270622e-03, -1.544374531065081e-06, -1.557004723352624e-05, -1.557004723352624e-05, 7.140352275024646e-03, 7.140351901058116e-03, 7.140352235547687e-03, 7.140351944802032e-03, 7.140352084869286e-03, 7.140352084869286e-03, 7.246118439170850e-03, 7.242118735545710e-03, 7.247785261098785e-03, 7.244593275074126e-03, 7.241338250486404e-03, 7.241338250486404e-03, -2.083580488316571e-02, -9.813496445152078e-03, -2.921434282909962e-02, -2.240703034890297e-02, -1.545625905451269e-02, -1.545625905451269e-02, -1.108237122290167e-02, -5.370724028725318e-02, -1.069218904149352e-02, 7.257927102158990e-03, -1.647646797332269e-02, -1.647646797332269e-02, -1.386489609034887e-06, -2.069426653094030e-06, -8.713852821931198e-07, -2.843632250225795e-03, -1.565455080312961e-06, -1.565455080312961e-06, -1.631825421413849e-02, -1.706070540844615e-02, -1.679273037429424e-02, -1.658738192710317e-02, -1.669000758524878e-02, -1.669000758524878e-02, -2.102290264221892e-02, -4.555677414918425e-02, -3.750808451295788e-02, -3.067012155609698e-02, -3.407389744440725e-02, -3.407389744440729e-02, -4.829037024000473e-03, -7.982286827812170e-02, -9.617773591230099e-02, -9.867805817918728e-02, -1.015467900263477e-01, -1.015467900263477e-01, -6.493216819065810e-02, -8.487464934673214e-04, -2.082401461491174e-03, -1.011415786081509e-01, -6.147555930856319e-03, -6.147555930856321e-03, -2.250266765802187e-05, -4.247701353661808e-08, -2.911674047986283e-07, -5.369921887604849e-03, -9.609790767276844e-07, -9.609790767276804e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05