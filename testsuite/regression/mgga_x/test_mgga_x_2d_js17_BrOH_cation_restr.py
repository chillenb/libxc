
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_2d_js17_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.506729218920690e+02, -1.506735996649901e+02, -1.506767404358051e+02, -1.506666539902729e+02, -1.506719209208022e+02, -1.506719209208022e+02, -9.643271649428282e+00, -9.643310889626457e+00, -9.644605601357334e+00, -9.645432328300611e+00, -9.643820293596535e+00, -9.643820293596535e+00, -9.297760865575587e-01, -9.304524939483838e-01, -9.553514308070320e-01, -9.639293793028444e-01, -9.595492260887099e-01, -9.595492260887099e-01, -3.513031962058434e-01, -3.437300790492283e-01, -1.104021152758833e+00, -4.110246568098370e-01, -3.762457459836843e-01, -3.762457459836844e-01, -9.540530487604447e-01, -9.238787570443464e-01, -6.467325391080995e-01, -1.154461942454772e+00, -1.004552451363986e+00, -1.004552451363986e+00, -1.788882223450502e+01, -1.789581551761246e+01, -1.788915453964364e+01, -1.789532786207985e+01, -1.789235958842099e+01, -1.789235958842099e+01, -4.293177457407692e+00, -4.328756723299979e+00, -4.282589155062948e+00, -4.313531366876213e+00, -4.318393819088096e+00, -4.318393819088096e+00, -7.139520652566188e-01, -7.555303523983737e-01, -6.495478044918486e-01, -6.306530993028227e-01, -7.220883683605903e-01, -7.220883683605903e-01, -4.969288374547382e-01, -4.561677370489060e-01, -4.994248516233298e-01, -3.985796136526893e+00, -4.220901432670137e-01, -4.220901432670137e-01, -1.134962036411798e+00, -1.101471251152986e+00, -6.908570211722187e-01, -5.222080961049554e-01, -8.191590242501988e-01, -8.191590242501993e-01, -6.785647053420738e-01, -6.689123757485632e-01, -6.703917536481070e-01, -6.728834263296299e-01, -6.714575701464016e-01, -6.714575701464016e-01, -6.571701650538042e-01, -6.267481451754990e-01, -6.160376099074187e-01, -6.127840260508701e-01, -6.126840279294569e-01, -6.126840279294569e-01, -8.105763477681845e-01, -4.614670441887182e-01, -4.450174242514739e-01, -4.207162609771194e-01, -4.245123813220675e-01, -4.245123813220674e-01, -5.660496449712580e-01, -6.762447358633155e-01, -6.076702224788980e-01, -3.668437834846030e-01, -4.546368203279462e-01, -4.546368203279462e-01, -9.095937720386493e-01, -9.780194799676903e-01, -1.017299386340371e+00, -4.672491885754794e-01, -7.833678804956468e-01, -7.833678804956468e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_2d_js17_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.303536958463646e+02, -2.303546464483519e+02, -2.303591883182087e+02, -2.303450386329655e+02, -2.303524079052357e+02, -2.303524079052357e+02, -1.451711391116492e+01, -1.451733348143803e+01, -1.452296073965303e+01, -1.451745304196242e+01, -1.451791417642696e+01, -1.451791417642696e+01, -9.096384203454558e-01, -9.048518913579997e-01, -7.855613155799459e-01, -8.016685913336616e-01, -8.012446407783914e-01, -8.012446407783914e-01, 3.617642810993362e-02, 3.027434010381394e-02, -1.221079911923721e+00, 9.200690952604434e-02, 6.871847019353576e-02, 6.871847019353591e-02, 3.131658361615714e-01, 3.034174645651900e-01, 2.088063904454608e-01, 3.829781436683241e-01, 3.317209748572668e-01, 3.317209748572670e-01, -2.757549694215972e+01, -2.758661839586253e+01, -2.757602177630518e+01, -2.758584017313835e+01, -2.758113136545895e+01, -2.758113136545895e+01, -5.258392713750583e+00, -5.342197879598075e+00, -5.166300639453603e+00, -5.240114327078268e+00, -5.349457994162844e+00, -5.349457994162844e+00, -8.394491698412909e-01, -1.122150070955676e+00, -7.166757639642927e-01, -9.014066751731578e-01, -8.794587881543290e-01, -8.794587881543290e-01, 1.395790006954954e-01, 7.780197901531759e-02, 1.434123251935630e-01, -6.162544903077050e+00, 1.069990386157366e-01, 1.069990386157366e-01, 3.908597678125850e-01, 3.709894641192245e-01, 2.295639218128831e-01, 1.614891504620893e-01, 2.749674968333014e-01, 2.749674968333018e-01, -1.044867897644118e+00, -9.872249774627024e-01, -1.007082518866805e+00, -1.024131925083639e+00, -1.015556855389829e+00, -1.015556855389829e+00, -1.013102337932366e+00, -5.276396493935352e-01, -6.447561515901334e-01, -7.757144209199689e-01, -7.075707665545437e-01, -7.075707665545437e-01, -1.200565566331406e+00, 3.419529806549174e-02, -3.892641748477442e-02, -2.540201623881467e-01, -1.321772577419056e-01, -1.321772577419058e-01, -4.485650565287401e-01, 2.178065547291933e-01, 1.931471837260207e-01, -2.752211817749446e-01, 1.327148498690412e-01, 1.327148498690414e-01, 3.064059756877394e-01, 3.683839446584303e-01, 3.358214620924219e-01, 1.377795533858869e-01, 2.612620648088444e-01, 2.612620648088441e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [9.613324943399545e-09, 9.613332146491450e-09, 9.613143283413296e-09, 9.613040626450891e-09, 9.613125862035221e-09, 9.613125862035221e-09, -2.521801109021824e-07, -2.497132821071564e-07, -1.920058877633328e-07, -2.875784471350914e-07, -2.479414645929797e-07, -2.479414645929797e-07, -1.676713135329060e-02, -1.684033316572795e-02, -1.851480467653195e-02, -1.785652470604713e-02, -1.801803217133595e-02, -1.801803217133595e-02, -3.824794299588860e+00, -3.685757520284359e+00, -8.606511569665645e-03, -8.864806429702570e+00, -6.324539026488596e+00, -6.324539026488595e+00, -2.362467754486141e+05, -2.017597516912780e+05, -6.501213331778225e+02, -1.420971469512990e+06, -7.010050587933984e+05, -7.010050587933990e+05, 2.587280017204991e-06, 2.608642728834168e-06, 2.588147818616228e-06, 2.607006544064644e-06, 2.598155569967735e-06, 2.598155569967735e-06, -1.268331835729400e-04, -1.219566119640837e-04, -1.313115027496606e-04, -1.269202502747601e-04, -1.219493735804943e-04, -1.219493735804943e-04, -2.958675169332498e-02, -6.034707566814426e-03, -4.295045471488028e-02, -2.373321008130500e-02, -2.697696570112332e-02, -2.697696570112332e-02, -2.746773576454983e+01, -3.171309392607278e+00, -3.586825439098878e+01, 1.949970465364126e-04, -1.766013197038103e+01, -1.766013197038103e+01, -3.560394410573430e+06, -1.586962248599584e+06, -5.540860563208135e+06, -1.384638099726691e+02, -2.577173914668260e+06, -2.577173914668259e+06, 3.215220884398192e-02, -9.252150710413068e-03, -8.673291981320519e-04, 9.747171869204223e-03, 3.879677939291220e-03, 3.879677939291220e-03, 7.779746508035620e-02, -6.516080953670414e-02, -5.396565353307543e-02, -4.079435197105551e-02, -4.809858404699357e-02, -4.809858404699357e-02, -5.694078988152699e-03, -1.401959701599440e+00, -7.590577711192980e-01, -2.931642359393622e-01, -4.803412987201527e-01, -4.803412987201528e-01, -9.371353705536944e-02, -7.355075460231070e+02, -2.674407636386250e+02, -3.617874666286288e-01, -6.555564901685544e+01, -6.555564901685544e+01, -7.301330382309397e+04, -1.735829727559361e+08, -1.264940490237693e+07, -8.009442775417243e+01, -3.484184114242590e+06, -3.484184114242604e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.553267260691982e-05, 3.553258093749827e-05, 3.553204548837003e-05, 3.553341198176818e-05, 3.553271415196999e-05, 3.553271415196999e-05, 4.381460128356554e-04, 4.381580613653724e-04, 4.384224963566119e-04, 4.378509442544193e-04, 4.381461888038365e-04, 4.381461888038365e-04, 3.485931498117155e-03, 3.481943087965869e-03, 3.381125843182638e-03, 3.355426237756558e-03, 3.368910658565333e-03, 3.368910658565333e-03, 1.302884423328609e-02, 1.308098459894820e-02, 2.925784558490185e-03, 1.376976806369470e-02, 1.367476763820436e-02, 1.367476763820436e-02, 3.696190772884709e-02, 3.680326841464662e-02, 1.987476640489502e-02, 4.273759660345333e-02, 4.187012096465234e-02, 4.187012096465236e-02, 2.725800390568679e-04, 2.727270917386009e-04, 2.725855062712771e-04, 2.727153351500730e-04, 2.726552193437002e-04, 2.726552193437002e-04, 8.042781258414783e-04, 8.002999521290157e-04, 8.016544887007781e-04, 7.981153393075916e-04, 8.034091676432226e-04, 8.034091676432226e-04, 4.749540487863248e-03, 5.384666345882434e-03, 5.117663506814214e-03, 6.011240054514420e-03, 4.754845503596330e-03, 4.754845503596330e-03, 1.470381569882949e-02, 1.109159358314971e-02, 1.523534900468885e-02, 1.202570085210261e-03, 1.511180171325915e-02, 1.511180171325915e-02, 4.881276231585562e-02, 4.445447456337350e-02, 6.958352248537812e-02, 1.801493382849299e-02, 5.646768313360769e-02, 5.646768313360766e-02, 6.940212058378999e-03, 6.072102608579075e-03, 6.301476446766478e-03, 6.541996386781751e-03, 6.414675910000288e-03, 6.414675910000288e-03, 7.748140148090674e-03, 5.224363072183512e-03, 5.384522406477146e-03, 5.717554240937010e-03, 5.524106904352445e-03, 5.524106904352445e-03, 4.979554696187549e-03, 9.609309812296078e-03, 8.786834585172289e-03, 7.871417905072781e-03, 8.348771238697905e-03, 8.348771238697905e-03, 5.775311820612195e-03, 1.973863441009645e-02, 1.814598463613503e-02, 8.902858110658744e-03, 1.754108685307929e-02, 1.754108685307930e-02, 3.193785189364660e-02, 9.076512184398713e-02, 6.284940997944853e-02, 1.779144544717011e-02, 6.056142697564759e-02, 6.056142697564763e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05