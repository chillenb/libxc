
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_lta_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.128952754591961e+01, -1.128927907125267e+01, -1.128829680053156e+01, -1.129197242026966e+01, -1.129002935971245e+01, -1.129002935971245e+01, -4.358038915675461e+00, -4.357135299872805e+00, -4.333438778274665e+00, -4.343397895387854e+00, -4.341877994230558e+00, -4.341877994230558e+00, -1.531846504618128e+00, -1.537890164908577e+00, -1.686859390600574e+00, -1.609715401001915e+00, -1.627277974811424e+00, -1.627277974811424e+00, -5.662379347398039e-01, -5.505933775624319e-01, -2.769442205168520e+00, -9.087268505820487e-01, -7.355472839776390e-01, -7.355472839776391e-01, -4.652112366308022e+00, -4.437335065173394e+00, -2.115494808267445e+00, -7.322632657606291e+00, -5.592785820871795e+00, -5.592785820871795e+00, -2.033279213320479e+00, -2.020568311544549e+00, -2.032544338152003e+00, -2.021331459170483e+00, -2.026911232672347e+00, -2.026911232672347e+00, -2.897860887713501e+00, -2.823699452148932e+00, -2.958770789780159e+00, -2.890053801619271e+00, -2.827111484608879e+00, -2.827111484608879e+00, -4.951844227871622e-01, -3.205539954317307e-01, -6.092161608563993e-01, -4.706485208851980e-01, -4.624302735144363e-01, -4.624302735144363e-01, -1.256001743652715e+00, -8.401068890312527e-01, -1.299222166456327e+00, -1.413373115370700e+00, -9.194211765510526e-01, -9.194211765510526e-01, -9.912887286346214e+00, -7.798341645524054e+00, -3.970033507523168e+00, -1.503744657611558e+00, -5.207796588284303e+00, -5.207796588284304e+00, -7.987860976436559e-02, -1.381667620776178e-01, -1.185014541047989e-01, -1.016109740637597e-01, -1.101467909402794e-01, -1.101467909402794e-01, -8.610035045794694e-02, -4.933438261860558e-01, -3.790485786190023e-01, -2.639795555608192e-01, -3.231087031861710e-01, -3.231087031861711e-01, -3.937127206319307e-01, -7.549036315078550e-01, -6.706805501266693e-01, -5.233015079234847e-01, -5.725995078389083e-01, -5.725995078389082e-01, -5.475203318563616e-01, -2.159883215611056e+00, -1.879129407519480e+00, -3.958425286784827e-01, -1.148852545970205e+00, -1.148852545970205e+00, -5.082495293540870e+00, -1.587370535988908e+01, -6.675039307950083e+00, -1.155669788043719e+00, -4.735073513710494e+00, -4.735073513710495e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_lta_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.767647893653400e-15, -4.416500484153709e-16, 3.627785363054931e-15, -1.099675748447041e-15, 1.157953415391983e-15, 1.157953415391983e-15, -1.660745315239309e-16, -1.383711694806763e-15, 8.255641922408821e-17, -2.000246051331189e-15, -2.029537323417937e-15, -2.029537323417937e-15, -8.799208704612039e-17, 4.035423623447324e-16, -3.201451835556327e-16, 1.245713711536114e-16, -6.334099622952036e-16, -6.334099622952036e-16, -1.119004416393027e-16, -2.045451955422456e-18, 1.065564587233562e-16, 1.925326941019882e-16, 1.996718893337375e-16, 7.948728257726906e-17, 2.285215130326012e-17, 9.635177752010719e-16, 2.919277316817485e-16, -5.905157622484089e-16, 1.916531136725103e-15, 1.916531136725103e-15, 3.168272358356987e-16, 7.196218318802251e-16, -1.768679822949739e-16, -1.561321751453608e-15, 4.811871249396068e-16, 4.811871249396068e-16, -2.580664354952171e-16, -1.139349448919553e-15, -1.724206128499310e-15, 1.928164897238936e-15, 3.166179080923791e-16, 3.166179080923791e-16, -1.724999836950678e-16, -2.960720765903640e-16, 8.012392743729314e-17, 1.911142517356198e-16, 3.285996964526458e-16, 3.285996964526458e-16, -1.503985104541200e-16, 1.432514557862997e-16, 2.670003904225724e-16, -3.008523417351922e-16, 4.603894918172519e-16, 4.603894918172519e-16, 8.589824418445690e-16, 3.397856382232460e-15, -1.043974282545482e-15, 1.144687250325259e-15, 1.263858785621231e-15, 1.287071432773499e-15, -2.045531625488788e-18, 1.917510127166849e-17, 4.134309337756053e-17, -2.456653837619348e-17, 6.353895130103646e-17, 6.353895130103646e-17, 1.026432240705671e-16, 7.294076215604362e-17, 1.773143362228233e-16, 3.627981919252030e-18, 1.006530548082748e-16, 1.095736509157641e-16, 3.554768874870536e-17, -2.764584255860751e-16, 8.372208478702760e-17, -2.025625373694552e-16, 2.109204963514621e-16, -1.777046004498796e-17, 6.909514361637791e-16, 4.319270634852731e-16, -7.348598046648181e-16, 2.339338401126389e-16, 3.721847802058907e-16, -2.638630659118190e-16, 6.164684631465577e-16, 9.068372525179497e-15, 2.264070677768368e-15, 5.486499190157899e-16, 6.589454220364450e-16, -1.533861385982329e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lta_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lta_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lta_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-8.825875265045854e-03, -8.825905367834865e-03, -8.826009591687592e-03, -8.825566034761473e-03, -8.825802570208536e-03, -8.825802570208536e-03, -4.368636203292307e-02, -4.368855613610533e-02, -4.374571880783806e-02, -4.371863540994334e-02, -4.372608130119246e-02, -4.372608130119246e-02, -1.949389057861342e-01, -1.949082945030297e-01, -1.943389803761970e-01, -1.954113808965342e-01, -1.951382592774727e-01, -1.951382592774727e-01, -6.901942064135862e-01, -6.872539713356310e-01, -1.498187822641276e-01, -7.539697264156848e-01, -7.286986915733307e-01, -7.286986915733306e-01, -5.058483605690128e+00, -4.925756565649992e+00, -1.646710156806373e+00, -6.816861374407309e+00, -6.144018528145742e+00, -6.144018528145742e+00, -3.905320805652962e-02, -3.910909742748250e-02, -3.905646970430957e-02, -3.910576856861741e-02, -3.908112880393718e-02, -3.908112880393718e-02, -7.445595647031637e-02, -7.453497750714283e-02, -7.436530940257845e-02, -7.444395648107592e-02, -7.455043360284140e-02, -7.455043360284140e-02, -2.858264335203455e-01, -2.994614016454723e-01, -2.876660466680950e-01, -2.980475191054795e-01, -2.872917178701824e-01, -2.872917178701824e-01, -9.142429429631226e-01, -6.246591975156673e-01, -9.610317626616098e-01, -9.039481115552365e-02, -8.766151302449817e-01, -8.766151302449819e-01, -7.676640262615288e+00, -6.825487200840795e+00, -9.881877855538775e+00, -1.247545004390757e+00, -8.030046381140709e+00, -8.030046381140709e+00, -4.493748947548040e-01, -3.935404062324188e-01, -4.083181785116441e-01, -4.237906609728863e-01, -4.155902074901612e-01, -4.155902074901611e-01, -4.502111583573357e-01, -3.262887995529627e-01, -3.366713536337517e-01, -3.566591357169862e-01, -3.447471111321827e-01, -3.447471111321827e-01, -2.745590783142068e-01, -5.383793627455962e-01, -4.833805925559001e-01, -4.174334941280401e-01, -4.532186370393750e-01, -4.532186370393750e-01, -3.382376515794536e-01, -1.692916068267386e+00, -1.389315552006257e+00, -4.600783762852613e-01, -1.113220088098513e+00, -1.113220088098513e+00, -3.821180143328799e+00, -1.537119445514640e+01, -1.094580089194741e+01, -1.165264836256940e+00, -8.716325646901700e+00, -8.716325646901705e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05