
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th_fl_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.926616258774007e+01, 1.926642149732492e+01, 1.926765247340670e+01, 1.926379880999619e+01, 1.926580666178607e+01, 1.926580666178607e+01, -3.907724860597666e+00, -3.907732682715946e+00, -3.908002275454258e+00, -3.908221711689311e+00, -3.907821829692173e+00, -3.907821829692173e+00, -8.675157882328403e-01, -8.667843349486950e-01, -8.491492145883717e-01, -8.544976954702165e-01, -8.547726484223650e-01, -8.547726484223650e-01, -3.736020282703935e-01, -3.765456446573945e-01, -9.794396524419035e-01, -3.252305062512162e-01, -3.647030426643560e-01, -3.647030426643559e-01, -7.757791660655416e-02, -7.949568352076707e-02, -1.754783722476115e-01, -5.970368921170186e-02, -6.968571986094300e-02, -6.968571986094300e-02, -5.509382743258135e+00, -5.510135553917937e+00, -5.509420244080365e+00, -5.510084788760902e+00, -5.509762835004452e+00, -5.509762835004452e+00, -2.243747347241943e+00, -2.259586447547546e+00, -2.232189950819874e+00, -2.246184924476903e+00, -2.258114602338908e+00, -2.258114602338908e+00, -7.845789333112588e-01, -8.344741624363630e-01, -7.420124743179211e-01, -7.629392268089735e-01, -7.946454101293701e-01, -7.946454101293701e-01, -2.729183294122510e-01, -3.737664802515709e-01, -2.632542020613842e-01, -2.202010922961783e+00, -2.946823789365962e-01, -2.946823789365962e-01, -5.273522365924908e-02, -5.905896189653780e-02, -5.195769683230140e-02, -2.189026274861496e-01, -5.783343838478745e-02, -5.783343838478745e-02, -7.873783307541773e-01, -7.840580054142479e-01, -7.852262482623562e-01, -7.861885376139738e-01, -7.857068535114012e-01, -7.857068535114012e-01, -7.716985229123871e-01, -6.924397254815470e-01, -7.150454713391226e-01, -7.375079196685874e-01, -7.260293490719633e-01, -7.260293490719633e-01, -8.650052704502152e-01, -4.228328996895991e-01, -4.680025377947971e-01, -5.516746115451673e-01, -5.070887433726019e-01, -5.070887433726019e-01, -6.544756034108750e-01, -1.718967409919037e-01, -1.981271683447501e-01, -5.394525047451648e-01, -2.448973444790876e-01, -2.448973444790876e-01, -9.120706914186226e-02, -3.119532811630300e-02, -4.475921889866954e-02, -2.377282495669906e-01, -5.553958170622917e-02, -5.553958170622914e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th_fl_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [5.021174151646211e+01, 5.021150635618859e+01, 5.021233151017980e+01, 5.021193496565810e+01, 5.021433547879239e+01, 5.021477390428922e+01, 5.020766959930243e+01, 5.020627863835189e+01, 5.021208752096635e+01, 5.020976001861855e+01, 5.021208752096635e+01, 5.020976001861855e+01, -5.115752088143185e+00, -5.115691870972763e+00, -5.115765886235496e+00, -5.115697124488484e+00, -5.116014387837690e+00, -5.116105249262416e+00, -5.116296075202812e+00, -5.116358017073888e+00, -5.115014053777456e+00, -5.116665113691252e+00, -5.115014053777456e+00, -5.116665113691252e+00, -1.087681825170210e+00, -1.091278725552970e+00, -1.086270117527271e+00, -1.090682313381571e+00, -1.067440501604675e+00, -1.061351962572107e+00, -1.070758152876302e+00, -1.072682120518618e+00, -1.098271231381976e+00, -1.040290312574649e+00, -1.098271231381976e+00, -1.040290312574649e+00, -4.328935059484664e-01, -4.419549942457871e-01, -4.357719276424338e-01, -4.462018303300025e-01, -1.223950205748114e+00, -1.259753933725622e+00, -3.771330014655477e-01, -3.797846545749705e-01, -4.431793550509427e-01, -3.222014338100778e-01, -4.431793550509427e-01, -3.222014338100778e-01, -8.841328109607638e-02, -9.097385252002424e-02, -9.037887669324270e-02, -9.334722276028837e-02, -1.995144552311141e-01, -2.045472593433294e-01, -6.949399642129757e-02, -6.894061654343191e-02, -8.378386563421777e-02, -6.411687981791254e-02, -8.378386563421777e-02, -6.411687981791256e-02, -6.876838389507132e+00, -6.875926090763697e+00, -6.877549228896816e+00, -6.876613141340715e+00, -6.876881686500971e+00, -6.875952423265953e+00, -6.877491588502139e+00, -6.876576545073886e+00, -6.877199595513726e+00, -6.876270716158998e+00, -6.877199595513726e+00, -6.876270716158998e+00, -2.971613002701215e+00, -2.971464631671299e+00, -2.993234409598360e+00, -2.992429763743227e+00, -2.953462446162079e+00, -2.958510826027691e+00, -2.972163801071725e+00, -2.977455020487323e+00, -2.997294235993659e+00, -2.984327415593472e+00, -2.997294235993659e+00, -2.984327415593472e+00, -9.775998323335864e-01, -9.755275119458454e-01, -1.044228143594012e+00, -1.044617642900657e+00, -9.052818109078470e-01, -9.311497961688795e-01, -9.344055253470895e-01, -9.588595664630638e-01, -1.009016739910790e+00, -9.683698602069883e-01, -1.009016739910790e+00, -9.683698602069886e-01, -3.150989211121673e-01, -3.165465304221272e-01, -4.372868716588763e-01, -4.384707999241401e-01, -3.003004123467249e-01, -3.079170731600442e-01, -2.916014907296305e+00, -2.914719595417818e+00, -3.364628384452308e-01, -3.461721494374708e-01, -3.364628384452308e-01, -3.461721494374708e-01, -6.058885187350207e-02, -6.171456483181583e-02, -6.823044406208358e-02, -6.872007392076772e-02, -5.938918694464301e-02, -6.104194000469672e-02, -2.521009000901683e-01, -2.528185371668644e-01, -5.999277995737368e-02, -6.979682819211354e-02, -5.999277995737368e-02, -6.979682819211354e-02, -9.788427069373760e-01, -9.818751385350233e-01, -9.743303250476201e-01, -9.773735513750842e-01, -9.759129570828418e-01, -9.789620413775203e-01, -9.772303581294272e-01, -9.802572666603733e-01, -9.765710620642155e-01, -9.796087618308167e-01, -9.765710620642155e-01, -9.796087618308167e-01, -9.578323207995487e-01, -9.603475623252203e-01, -8.506873017673259e-01, -8.534697545057430e-01, -8.810018590544596e-01, -8.839999602265564e-01, -9.115680331101844e-01, -9.140622622113797e-01, -8.960302896920741e-01, -8.986013356585653e-01, -8.960302896920741e-01, -8.986013356585653e-01, -1.085324496494685e+00, -1.086798575814047e+00, -4.983139356815392e-01, -4.999803751001709e-01, -5.548356620220088e-01, -5.582305984710335e-01, -6.637910817781101e-01, -6.662487385857556e-01, -6.069835701070887e-01, -6.068466529733092e-01, -6.069835701070887e-01, -6.068466529733090e-01, -7.991604498494904e-01, -8.031981259916359e-01, -1.978004383125697e-01, -1.984176810818068e-01, -2.265596588139192e-01, -2.300137446911656e-01, -6.457482561134988e-01, -6.521770693187453e-01, -2.788962643574792e-01, -2.861401735009908e-01, -2.788962643574792e-01, -2.861401735009908e-01, -1.045548708314983e-01, -1.063054375655518e-01, -3.625061654268589e-02, -3.629090279926917e-02, -5.112956780675613e-02, -5.266895068188011e-02, -2.733467923386015e-01, -2.754347237957105e-01, -5.845483792197638e-02, -6.696157688953415e-02, -5.845483792197635e-02, -6.696157688953412e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th_fl_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05