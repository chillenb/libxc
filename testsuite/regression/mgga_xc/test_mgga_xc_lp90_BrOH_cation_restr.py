
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data

# test_mgga_xc_lp90_BrOH_cation_restr_1_zk() not generated due to NaN

# test_mgga_xc_lp90_BrOH_cation_restr_1_vrho() not generated due to NaN


def test_mgga_xc_lp90_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.541568324533171e-10, -6.541493552403181e-10, -6.541138070704677e-10, -6.542251043759161e-10, -6.541671120269878e-10, -6.541671120269878e-10, -1.014792288868781e-06, -1.014783489287707e-06, -1.014480258735946e-06, -1.014233519850619e-06, -1.014683467527390e-06, -1.014683467527390e-06, -7.437012362704908e-04, -7.469929228319390e-04, -8.319908100943070e-04, -8.049084159413682e-04, -8.105745005745812e-04, -8.105745005745812e-04, -1.677266870605841e-01, -1.579370776848260e-01, -4.020522238038700e-04, -5.050269529076052e-01, -3.176021224324803e-01, -3.176021224324804e-01, -1.143150774642284e+05, -9.314210257448621e+04, -1.005166025800998e+02, -1.027560017220101e+06, -4.121444163556091e+05, -4.121444163556091e+05, -2.002082814375229e-07, -2.000603727330063e-07, -2.002009115688695e-07, -2.000703428882819e-07, -2.001335900435752e-07, -2.001335900435752e-07, -1.019480576397535e-05, -9.903588141853116e-06, -1.041418240404464e-05, -1.014936045886736e-05, -9.930590299217430e-06, -9.930590299217430e-06, -1.270791488875978e-03, -9.121776546241434e-04, -1.733952490514081e-03, -1.484818989030349e-03, -1.191981528299620e-03, -1.191981528299620e-03, -2.173973102041336e+00, -1.667114599844230e-01, -2.968016163746320e+00, -1.101544452235496e-05, -1.146138634327008e+00, -1.146138634327008e+00, -2.899448528064258e+06, -1.125079174813354e+06, -3.291080354281168e+06, -1.450598352191887e+01, -1.562557978552480e+06, -1.562557978552480e+06, -1.246388874650872e-03, -1.275424130895437e-03, -1.265111629913918e-03, -1.256695743773509e-03, -1.260899495193452e-03, -1.260899495193452e-03, -1.391322940014323e-03, -2.563259462166191e-03, -2.131519454979804e-03, -1.789467408109263e-03, -1.954818919638120e-03, -1.954818919638120e-03, -7.550143561468679e-04, -6.541057969293687e-02, -3.143789303033171e-02, -1.032374914917327e-02, -1.805423838661189e-02, -1.805423838661189e-02, -3.568772192337155e-03, -1.197741086830259e+02, -3.466732107667493e+01, -1.195446055411286e-02, -5.517791275842375e+00, -5.517791275842374e+00, -2.881455628697400e+04, -2.203693350204365e+08, -1.135174342166843e+07, -7.096326578323077e+00, -2.131485479290419e+06, -2.131485479290428e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_lp90_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.264695156790174e-05, 1.264691282060177e-05, 1.264672860364657e-05, 1.264730534120612e-05, 1.264700483663832e-05, 1.264700483663832e-05, 8.466685247068036e-05, 8.466666652949912e-05, 8.466025832121208e-05, 8.465504289825267e-05, 8.466455291614381e-05, 8.466455291614381e-05, 4.452460137934207e-04, 4.457391339266607e-04, 4.579423930325041e-04, 4.541599533691275e-04, 4.549591116900226e-04, 4.549591116900226e-04, 1.728736701593672e-03, 1.702922816980106e-03, 3.816242761598568e-04, 2.277594709595485e-03, 2.028111013319541e-03, 2.028111013319541e-03, 4.970302417918278e-02, 4.722184121728022e-02, 8.557916859327561e-03, 8.606230667455105e-02, 6.848919285487624e-02, 6.848919285487624e-02, 5.606205831907257e-05, 5.605150077240937e-05, 5.606153240510047e-05, 5.605221261408726e-05, 5.605672766773350e-05, 5.605672766773350e-05, 1.516053463862180e-04, 1.505027671936188e-04, 1.524204302580649e-04, 1.514348547794349e-04, 1.506060150279852e-04, 1.506060150279852e-04, 5.092237423495495e-04, 4.686252718550430e-04, 5.504555323271195e-04, 5.294752359413667e-04, 5.011199873644127e-04, 5.011199873644127e-04, 3.281167035375734e-03, 1.726113048461514e-03, 3.546849462434591e-03, 1.545902418816406e-04, 2.795745813167958e-03, 2.795745813167958e-03, 1.115427531846991e-01, 8.803534779562261e-02, 1.151323237413502e-01, 5.274230172927461e-03, 9.556982604011045e-02, 9.556982604011045e-02, 5.067557959413530e-04, 5.096882403885656e-04, 5.086525074402299e-04, 5.078025578425939e-04, 5.082276411526812e-04, 5.082276411526812e-04, 5.209174314026955e-04, 6.070788315264039e-04, 5.796697674653842e-04, 5.548185471273855e-04, 5.672391759558902e-04, 5.672391759558902e-04, 4.469340183071958e-04, 1.365883785750512e-03, 1.137081944339143e-03, 8.604926537727030e-04, 9.897101927937679e-04, 9.897101927937679e-04, 6.595407844950138e-04, 8.941324407229716e-03, 6.557992070047187e-03, 8.926703830516899e-04, 4.141786624513636e-03, 4.141786624513635e-03, 3.521727791076647e-02, 3.293466588397041e-01, 1.569023490827754e-01, 4.410747045782003e-03, 1.032839277949216e-01, 1.032839277949217e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_lp90_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05