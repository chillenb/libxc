
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpss_gaussian_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.771284813680278e-02, -4.771339015203847e-02, -4.771436849447267e-02, -4.770630348616414e-02, -4.771073329542933e-02, -4.771073329542933e-02, -4.180384615737967e-02, -4.180882180559695e-02, -4.192437295891271e-02, -4.172754754210879e-02, -4.181153336413154e-02, -4.181153336413154e-02, -2.776586075010862e-02, -2.753893861650485e-02, -2.211925539905948e-02, -2.243697658992485e-02, -2.255399743747880e-02, -2.255399743747880e-02, -7.484007786831430e-03, -8.095764142238735e-03, -3.005598799275944e-02, -2.317574613592221e-03, -4.089866793405334e-03, -4.089866793405333e-03, -6.979260682921377e-09, -9.308918328755660e-09, -1.039429365636090e-05, -4.722047282225437e-10, -1.775147164816801e-09, -1.775147168859411e-09, -5.045710652353631e-02, -5.056741248193014e-02, -5.045803179122788e-02, -5.055552619940792e-02, -5.051514211991568e-02, -5.051514211991568e-02, -2.297688380359029e-02, -2.347735897147641e-02, -2.203399905913350e-02, -2.247312125633386e-02, -2.372635725442509e-02, -2.372635725442509e-02, -3.836076687017409e-02, -5.507118623747214e-02, -3.619430228378180e-02, -5.101140468224478e-02, -3.998978252390782e-02, -3.998978252390782e-02, -4.748651141544485e-04, -3.887788341623976e-03, -3.691101733859214e-04, -7.151150075700284e-02, -1.276162366786151e-03, -1.276162366786151e-03, -1.681884578254013e-10, -4.771370422294964e-10, -8.825430955922371e-10, -9.542497813686718e-05, -9.570940829947824e-10, -9.570940749253926e-10, -5.560105504458351e-02, -4.763882029003909e-02, -4.965801881773658e-02, -5.184727154706167e-02, -5.068094745282712e-02, -5.068094745282711e-02, -6.086381140725119e-02, -2.805876572875166e-02, -3.362163966209303e-02, -4.020447559207625e-02, -3.676618941133149e-02, -3.676618941133149e-02, -5.520104008130137e-02, -6.917217504154714e-03, -1.150435138900250e-02, -2.429759960972572e-02, -1.737726825054464e-02, -1.737726825054466e-02, -2.736869439469190e-02, -8.020025177333460e-06, -2.974613799930040e-05, -2.868733003536878e-02, -3.175566649371603e-04, -3.175566649371557e-04, -2.476342011164112e-08, -3.964450525217989e-12, -7.886078331745399e-11, -2.476103089406131e-04, -8.551934505397901e-10, -8.551934552091174e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpss_gaussian_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.133323249016094e-01, -1.133332585294516e-01, -1.133351959860774e-01, -1.133213122810033e-01, -1.133289018474193e-01, -1.133289018474193e-01, -1.064033551954388e-01, -1.064087334783885e-01, -1.065283293872341e-01, -1.062685278631970e-01, -1.063847513756560e-01, -1.063847513756560e-01, -7.737100404342000e-02, -7.708790980225655e-02, -6.916027953123487e-02, -6.952564855656940e-02, -6.977310033976349e-02, -6.977310033976349e-02, -2.934136986760031e-02, -3.106736006840802e-02, -8.272306877757285e-02, -1.144540468836138e-02, -1.852636614802775e-02, -1.852636614802775e-02, -4.246230494485124e-08, -5.663579456139812e-08, -6.156717269755939e-05, -2.926382351406453e-09, -1.092011715024018e-08, -1.092011713787792e-08, -1.155847964192792e-01, -1.158057598891356e-01, -1.155924360877383e-01, -1.157875692134975e-01, -1.156979440558004e-01, -1.156979440558004e-01, -7.125197911830862e-02, -7.191525872052432e-02, -6.938779206886264e-02, -6.998888497328631e-02, -7.252606234161033e-02, -7.252606234161033e-02, -7.895039659206263e-02, -8.292810877038059e-02, -7.847768211711403e-02, -8.098606085841253e-02, -7.979705607158319e-02, -7.979705607158319e-02, -2.617054411526841e-03, -1.772148009985032e-02, -2.057028926501176e-03, -1.174995506743234e-01, -6.591948542642611e-03, -6.591948542642612e-03, -1.070554763821668e-09, -2.994086506261399e-09, -5.495204983420631e-09, -5.520285062360533e-04, -5.996575686358195e-09, -5.996575677673935e-09, -7.879099889964407e-02, -7.970695918859191e-02, -8.008775004491589e-02, -8.001671677735608e-02, -8.011613988762437e-02, -8.011613988762435e-02, -7.379770527221632e-02, -6.601449601523926e-02, -7.011702081855849e-02, -7.454519278397441e-02, -7.231803866617301e-02, -7.231803866617301e-02, -8.488735706834588e-02, -2.821787668923126e-02, -4.084515737190437e-02, -6.240261590056611e-02, -5.231418215605557e-02, -5.231418215605559e-02, -6.695102337286711e-02, -4.729236652877883e-05, -1.746387561370932e-04, -6.434159647347026e-02, -1.772052225786020e-03, -1.772052225786037e-03, -1.537592718695256e-07, -2.590178750404018e-11, -4.895266501463189e-10, -1.385029563876194e-03, -5.335753929149929e-09, -5.335753932031777e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.296930888715393e-10, 2.297011247362500e-10, 2.297271449376944e-10, 2.296083252697949e-10, 2.296719430075454e-10, 2.296719430075454e-10, 1.018045752755748e-06, 1.018261472626247e-06, 1.023078155339284e-06, 1.013040103571900e-06, 1.017831923482847e-06, 1.017831923482847e-06, 1.701274658385958e-03, 1.687518728925727e-03, 1.346204385894291e-03, 1.305834569786862e-03, 1.334097979474730e-03, 1.334097979474730e-03, 1.291526945207731e-01, 1.355129968224990e-01, 9.566444239974965e-04, 1.058544972035709e-01, 1.309399855294181e-01, 1.309399855294180e-01, 4.258598242810676e-03, 5.017343735326230e-03, 2.564952950296039e-02, 1.504792538941231e-03, 3.145220751667682e-03, 3.145220742753210e-03, 4.473237989971616e-07, 4.525255622139251e-07, 4.476247666869505e-07, 4.522134883952560e-07, 4.499173036877922e-07, 4.499173036877922e-07, 5.019413280832713e-06, 4.996013822843887e-06, 4.756917037169848e-06, 4.736737707083452e-06, 5.138981547484474e-06, 5.138981547484474e-06, 5.662232708479737e-03, 8.708782485770727e-03, 7.182679110045088e-03, 1.187185049079478e-02, 5.851910480549133e-03, 5.851910480549133e-03, 6.132640013319850e-02, 5.087517126187465e-02, 6.272281451675092e-02, 6.177805241319419e-05, 1.146515730010383e-01, 1.146515730010384e-01, 1.469809164188035e-03, 1.849996574961903e-03, 1.851034140110697e-02, 6.176107652962057e-02, 8.055826173160268e-03, 8.055826182736417e-03, 5.103632116101535e-02, 2.993079665704744e-02, 3.610569825362319e-02, 4.253026408878623e-02, 3.916257655710702e-02, 3.916257655710700e-02, 2.974429940758812e-02, 6.660076227444315e-03, 9.085995530083001e-03, 1.465315884109969e-02, 1.122257502080894e-02, 1.122257502080894e-02, 6.537962168809867e-03, 3.591481487455160e-02, 3.172665483451572e-02, 2.912388368589619e-02, 3.092013720336398e-02, 3.092013720336400e-02, 9.427908753204572e-03, 2.096693010763055e-02, 3.211242431970401e-02, 4.769213781859751e-02, 1.069939411499782e-01, 1.069939411499767e-01, 5.290407183705933e-03, 2.005669264814965e-03, 2.523650901537026e-03, 9.780429263448975e-02, 1.003131687480924e-02, 1.003131683711699e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.476926021364874e-05, -1.477043328569367e-05, -1.477640621929448e-05, -1.475909495627229e-05, -1.476805324258744e-05, -1.476805324258744e-05, 4.901944663798252e-06, 4.896074732397716e-06, 4.754033435441905e-06, 4.961454208945422e-06, 4.876836248412564e-06, 4.876836248412564e-06, 1.079198239762640e-04, 1.098095161843578e-04, 1.531233796018755e-04, 1.761162889668668e-04, 1.667544772862722e-04, 1.667544772862722e-04, 3.841373684197720e-03, 4.033604434519555e-03, 1.422796733430949e-05, 7.926928086751088e-04, 1.690598711709504e-03, 1.690598711709502e-03, 4.270259306058494e-09, 5.774380277135050e-09, 4.327769497722964e-06, 1.999179232506251e-10, 9.722567792053433e-10, 9.722567903079826e-10, -6.641982659253480e-04, -6.749562456794595e-04, -6.650167717593685e-04, -6.745014652286235e-04, -6.694581929282707e-04, -6.694581929282707e-04, 9.586520643360562e-05, 9.441870976997835e-05, 9.906286891198361e-05, 9.836958620492330e-05, 9.331211014535158e-05, 9.331211014535158e-05, -1.446981141068787e-03, -2.395763958405635e-03, -2.449152007691915e-04, -5.857634355691258e-04, -2.035859435813849e-03, -2.035859435813849e-03, 1.674027252792565e-04, 1.434173637998151e-03, 1.309327125403513e-04, -5.281626660563506e-05, 6.165235634273432e-04, 6.165235634273434e-04, 3.329441228889948e-11, 1.491215714726766e-10, 9.491702459839428e-10, 3.883266113945220e-05, 5.506978580688209e-10, 5.506978502481262e-10, -7.734457444537504e-02, -6.146135862654926e-02, -7.119253509866720e-02, -7.812978327811376e-02, -7.494944411217043e-02, -7.494944411217036e-02, -1.096015140056052e-02, -9.619027844867933e-04, -5.839253883351447e-03, -1.697268881067848e-02, -1.002750412304571e-02, -1.002750412304570e-02, -1.318428606049210e-03, 2.226211174662566e-03, 2.691619178809596e-03, 1.272792070114984e-03, 2.631570046631941e-03, 2.631570046631959e-03, 5.308729389042627e-04, 3.669608570785945e-06, 1.139547580402530e-05, -1.417020578835496e-05, 1.567550668297139e-04, 1.567550668297113e-04, 7.025313839368988e-09, 3.411869083830528e-13, 6.110834972581864e-11, 1.390776056405386e-04, 6.463132764916026e-10, 6.463132899966529e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05