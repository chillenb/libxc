
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3p86_nwchem_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.689904817675059e+01, -1.689907134004680e+01, -1.689922680685987e+01, -1.689888110179270e+01, -1.689905478252118e+01, -1.689905478252118e+01, -2.858442509516391e+00, -2.858417839546663e+00, -2.857916188852150e+00, -2.859318874508326e+00, -2.858480415973435e+00, -2.858480415973435e+00, -6.123350016895539e-01, -6.120737986865989e-01, -6.069435672306790e-01, -6.110767137098828e-01, -6.112703891586466e-01, -6.112703891586466e-01, -1.991536398685327e-01, -2.007620644419515e-01, -7.021281477663937e-01, -1.690562148569436e-01, -1.880890444213690e-01, -1.880890444213690e-01, -5.586561920491014e-02, -5.619970298750261e-02, -1.059473633259793e-01, -4.898584637955963e-02, -5.058018946857357e-02, -5.058018946857355e-02, -4.129996772315349e+00, -4.129686122025547e+00, -4.129990235294676e+00, -4.129715851651089e+00, -4.129834905679552e+00, -4.129834905679552e+00, -1.740567441884892e+00, -1.749048459409565e+00, -1.740388987683638e+00, -1.747852901509629e+00, -1.745388330348052e+00, -1.745388330348052e+00, -5.301873611830649e-01, -5.618702857602940e-01, -4.962345812886656e-01, -5.052597516602098e-01, -5.370056669761715e-01, -5.370056669761716e-01, -1.432056360145424e-01, -2.085795918931368e-01, -1.377809500165957e-01, -1.552772801392771e+00, -1.491883843246451e-01, -1.491883843246451e-01, -4.374086378157671e-02, -4.730738504816438e-02, -3.187588522990255e-02, -1.166843959401789e-01, -3.852838519722041e-02, -3.852838519722042e-02, -5.223024857244752e-01, -5.212568851188332e-01, -5.215984799404673e-01, -5.218964365514666e-01, -5.217445032704465e-01, -5.217445032704465e-01, -5.093019674446523e-01, -4.616494258956433e-01, -4.746055370822037e-01, -4.877490814145181e-01, -4.809135317949320e-01, -4.809135317949320e-01, -5.865141029998159e-01, -2.465656167448863e-01, -2.798900999074432e-01, -3.407288433020393e-01, -3.077821268421617e-01, -3.077821268421617e-01, -4.290997970968221e-01, -1.061664327264031e-01, -1.148907898859901e-01, -3.266096266112520e-01, -1.227846894707520e-01, -1.227846894707520e-01, -6.414636481357543e-02, -2.604973843883247e-02, -3.511721245277573e-02, -1.202315849470908e-01, -3.623330412505860e-02, -3.623330412505858e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3p86_nwchem_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.037106420100124e+01, -2.037104018001531e+01, -2.037115047641931e+01, -2.037110324231081e+01, -2.037141324046400e+01, -2.037148124241605e+01, -2.037043408713587e+01, -2.037023905070876e+01, -2.037111280629610e+01, -2.037072754817864e+01, -2.037111280629610e+01, -2.037072754817864e+01, -3.388013523688713e+00, -3.388113363870373e+00, -3.388051115470708e+00, -3.388154036956187e+00, -3.388987091161120e+00, -3.389187503909523e+00, -3.387893160552608e+00, -3.388092166648455e+00, -3.387459846571976e+00, -3.388917140722072e+00, -3.387459846571976e+00, -3.388917140722072e+00, -7.214543356198770e-01, -7.247237231728756e-01, -7.199710552599975e-01, -7.239710964610047e-01, -6.966914949428393e-01, -6.914992563572762e-01, -6.988438398931511e-01, -7.005189645126005e-01, -7.239660759643560e-01, -6.735262069676778e-01, -7.239660759643560e-01, -6.735262069676778e-01, -2.144676363350208e-01, -2.188764820714748e-01, -2.177686215866380e-01, -2.228676359844327e-01, -8.197195408374618e-01, -8.456473403500034e-01, -1.610201280278258e-01, -1.622231822038400e-01, -1.988206669103741e-01, -1.571144088651361e-01, -1.988206669103740e-01, -1.571144088651360e-01, -2.410721002378487e-02, -2.430550518742924e-02, -2.474888354028532e-02, -2.495824031187506e-02, -5.224293330703583e-02, -5.259542809994761e-02, -1.770376396750357e-02, -1.768724991611765e-02, -2.057561281790926e-02, -1.773124783557442e-02, -2.057561281790924e-02, -1.773124783557441e-02, -5.128295700909025e+00, -5.127114864824165e+00, -5.130555418602763e+00, -5.129309919304916e+00, -5.128416742829085e+00, -5.127193251305306e+00, -5.130364478246567e+00, -5.129178278582836e+00, -5.129451686482638e+00, -5.128219576790573e+00, -5.129451686482638e+00, -5.128219576790573e+00, -1.866700747737377e+00, -1.866613995906423e+00, -1.882488043303006e+00, -1.882002260507667e+00, -1.848916512019887e+00, -1.852888044829439e+00, -1.862570164760715e+00, -1.866677776578426e+00, -1.887925789099078e+00, -1.878002089431029e+00, -1.887925789099078e+00, -1.878002089431029e+00, -6.578151263769779e-01, -6.567195018890714e-01, -7.208049183629930e-01, -7.212910099584693e-01, -6.035586752204604e-01, -6.193237010614187e-01, -6.379610701647390e-01, -6.510928392053902e-01, -6.797364662736438e-01, -6.564852446927536e-01, -6.797364662736438e-01, -6.564852446927537e-01, -1.028946156468892e-01, -1.033300557189316e-01, -2.065856973789964e-01, -2.071710500699608e-01, -9.314554778662858e-02, -9.579389858451690e-02, -2.006148203312251e+00, -2.005442357580284e+00, -1.289437731686685e-01, -1.319810490069071e-01, -1.289437731686685e-01, -1.319810490069071e-01, -1.476667177411911e-02, -1.506673650508793e-02, -1.716522917934054e-02, -1.732124646808553e-02, -1.277151661467353e-02, -1.282857264514906e-02, -6.265258552426577e-02, -6.295671693774493e-02, -1.459339278576277e-02, -1.550663811589969e-02, -1.459339278576279e-02, -1.550663811589969e-02, -6.740519066618375e-01, -6.758284340420625e-01, -6.679051488532664e-01, -6.697409333654863e-01, -6.700789364766044e-01, -6.719136200587020e-01, -6.718860635240863e-01, -6.736678773056516e-01, -6.709842169765633e-01, -6.727920097031024e-01, -6.709842169765633e-01, -6.727920097031024e-01, -6.584062875810958e-01, -6.598027867738308e-01, -5.517348708731425e-01, -5.534930229953310e-01, -5.822643575371900e-01, -5.842104895232401e-01, -6.130207167265619e-01, -6.145076114870115e-01, -5.974955183864581e-01, -5.990276026424611e-01, -5.974955183864581e-01, -5.990276026424611e-01, -7.513803203035525e-01, -7.526259296263551e-01, -2.580729240614449e-01, -2.589591044514123e-01, -3.085953071250560e-01, -3.105596747261942e-01, -4.070396785240903e-01, -4.084581429829354e-01, -3.556682573941541e-01, -3.556773281281441e-01, -3.556682573941542e-01, -3.556773281281441e-01, -5.126570489854002e-01, -5.152556335795972e-01, -5.311481739374444e-02, -5.314942435492973e-02, -5.483899453140154e-02, -5.525192240462507e-02, -3.972042942475420e-01, -4.009637514995153e-01, -8.262644318375037e-02, -8.415718036503798e-02, -8.262644318375041e-02, -8.415718036503801e-02, -3.005062004055839e-02, -3.010748559221288e-02, -7.361714156669202e-03, -6.949337860602103e-03, -1.147938935520876e-02, -1.158782789396908e-02, -7.689938995964123e-02, -7.756615919338898e-02, -1.328221048144722e-02, -1.474950536853588e-02, -1.328221048144723e-02, -1.474950536853587e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3p86_nwchem_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.062187680536546e-08, -3.228412736824136e-10, -1.062193677212224e-08, -1.062181743441766e-08, -3.228402969419139e-10, -1.062189346255032e-08, -1.062149525564217e-08, -3.228301010432921e-10, -1.062147004763016e-08, -1.062217085653245e-08, -3.228447294126412e-10, -1.062232812039123e-08, -1.062184650274588e-08, -3.228379071744598e-10, -1.062188993451801e-08, -1.062184650274588e-08, -3.228379071744598e-10, -1.062188993451801e-08, -1.332380067443026e-05, 9.882426136229381e-07, -1.332900052582700e-05, -1.332395319750759e-05, 9.887050576800609e-07, -1.332954342751685e-05, -1.333192556899447e-05, 9.992776717892891e-07, -1.333508465839725e-05, -1.330980889969161e-05, 9.797537129189278e-07, -1.331391516418636e-05, -1.333109261643371e-05, 9.886452081975237e-07, -1.332003025206414e-05, -1.333109261643371e-05, 9.886452081975237e-07, -1.332003025206414e-05, -7.061483171642883e-03, 3.157389543744275e-03, -7.069653693894689e-03, -7.073793209102514e-03, 3.141588560554956e-03, -7.083344577433248e-03, -7.370432032008802e-03, 2.709936831111437e-03, -7.368792264576535e-03, -7.160239807290397e-03, 2.638315205881254e-03, -7.164797470192062e-03, -7.264152521869912e-03, 2.676184624575107e-03, -7.080922712850543e-03, -7.264152521869912e-03, 2.676184624575107e-03, -7.080922712850543e-03, -9.056311420798617e-01, 5.131795695701475e-01, -7.930401741049173e-01, -8.839514785253253e-01, 5.122620585167450e-01, -7.574716548527314e-01, -4.171965714718655e-03, 1.667953175927246e-03, -3.847455570843325e-03, -1.937791573706799e+00, 6.969696704844084e-01, -1.837677955065038e+00, -6.959603576363966e-01, 5.902065927012461e-01, -5.298859896474635e+00, -6.959603576363963e-01, 5.902065927012460e-01, -5.298859896474640e+00, -1.601673132872440e+04, -8.801724209675806e+00, -1.341767679668728e+04, -1.421579106446383e+04, -1.176252995576329e+01, -1.165362933000177e+04, -1.005643766969105e+02, -1.658373864244567e+01, -8.773727824723186e+01, -6.750579431967100e+04, -8.391960425752988e-01, -7.031049712854567e+04, -2.221665174583283e+04, -4.760263297495377e+00, -1.460232614826168e+05, -2.221665174583286e+04, -4.760263297495415e+00, -1.460232614826167e+05, -2.994178428465465e-06, 3.159453464004554e-07, -2.997012238140220e-06, -2.994437035982406e-06, 3.199934978278792e-07, -2.997196684146481e-06, -2.994190183083568e-06, 3.161088994050259e-07, -2.996982549070463e-06, -2.994362404968065e-06, 3.196822924928796e-07, -2.997202396556876e-06, -2.994342966132796e-06, 3.180042363884568e-07, -2.997095542914424e-06, -2.994342966132796e-06, 3.180042363884568e-07, -2.997095542914424e-06, -1.036065598351259e-04, 3.366858137744414e-06, -1.036254702981903e-04, -1.013273609059596e-04, 3.541907566732560e-06, -1.014193239709044e-04, -1.042546397500841e-04, 2.630424073294445e-06, -1.042147422347498e-04, -1.022768721453063e-04, 2.792606932933811e-06, -1.021938856552583e-04, -1.018963789580154e-04, 3.830326196520517e-06, -1.022522306809744e-04, -1.018963789580154e-04, 3.830326196520517e-06, -1.022522306809744e-04, -1.307237675577738e-02, 9.015492847758402e-03, -1.327149311232741e-02, -1.072282068345118e-02, 1.014956536066282e-02, -1.074707294844173e-02, -1.893272985411052e-02, 1.188929280105433e-02, -1.607356896817178e-02, -1.871039739152909e-02, 1.558246973853453e-02, -1.537239528631728e-02, -1.106319583890957e-02, 8.860540782460589e-03, -1.438921025791682e-02, -1.106319583890957e-02, 8.860540782460589e-03, -1.438921025791682e-02, -5.678777379140400e+00, 2.128416359971166e-01, -5.573465013016285e+00, -7.267189197076560e-01, 2.820420153571364e-01, -7.138389431195535e-01, -7.821242941722805e+00, -9.232854113920727e-03, -6.761368856675522e+00, -1.489264078344348e-04, 8.863053580097506e-05, -1.493083460677277e-04, -3.802771612581088e+00, 9.171558989922359e-01, -3.516727804914926e+00, -3.802771612581088e+00, 9.171558989922359e-01, -3.516727804914926e+00, -1.699414895564240e+05, -7.611841527052976e-01, -1.460800053477323e+05, -7.954741270882060e+04, -1.490811723083083e+00, -7.449580836918198e+04, -2.992594116709057e+05, -3.752539179419557e+02, -2.614614200749006e+05, -2.598867095618494e+01, -4.179011635322388e+00, -2.531325482250096e+01, -2.314231223241307e+05, -5.281233094143472e+01, -9.243129419502527e+04, -2.314231223241305e+05, -5.281233094143447e+01, -9.243129419502524e+04, -1.514977394523795e-02, 1.722113290849174e-02, -1.482514712734341e-02, -1.506845171828258e-02, 1.460739951458409e-02, -1.477057717902132e-02, -1.511337710822971e-02, 1.533630662307876e-02, -1.481028155386499e-02, -1.514710697867801e-02, 1.606590201721765e-02, -1.483036705729266e-02, -1.513184496015549e-02, 1.568363790639911e-02, -1.482203597629420e-02, -1.513184496015549e-02, 1.568363790639911e-02, -1.482203597629420e-02, -1.689064187885264e-02, 2.098893021379511e-02, -1.653928349241991e-02, -2.340013027084530e-02, 1.368602617894253e-02, -2.300458787571802e-02, -2.116462010175898e-02, 1.436092067730267e-02, -2.079790826898035e-02, -1.937302962356695e-02, 1.552674071178302e-02, -1.904029348453406e-02, -2.026458359301788e-02, 1.489560636469810e-02, -1.990628215518290e-02, -2.026458359301788e-02, 1.489560636469810e-02, -1.990628215518290e-02, -8.899593179328997e-03, 8.111017706436343e-03, -8.879980079260143e-03, -3.402472283428623e-01, 1.542208503466373e-01, -3.334811247779642e-01, -1.954381675866451e-01, 1.041643874318326e-01, -1.894165767090548e-01, -8.614972596633862e-02, 5.995134900497334e-02, -8.403622522256657e-02, -1.283998503869859e-01, 8.082587159628378e-02, -1.291917921655873e-01, -1.283998503869860e-01, 8.082587159628382e-02, -1.291917921655874e-01, -3.207446226338830e-02, 1.961668716112302e-02, -3.126297804145279e-02, -1.028955496144076e+02, -1.648975802716597e+01, -1.011480021461397e+02, -4.615121667155446e+01, -9.356814775292905e+00, -4.282438737490920e+01, -1.088524898579736e-01, 8.489087749794065e-02, -1.017894061417520e-01, -1.347841739287836e+01, -9.725137301966187e-02, -1.236321485283449e+01, -1.347841739287836e+01, -9.725137301966190e-02, -1.236321485283449e+01, -5.378566090691822e+03, -1.094133493679440e+01, -4.886620788694335e+03, -5.023961750865807e+06, -1.676145706781664e+00, -5.846009238659542e+06, -5.637411380854080e+05, -3.865859031857028e+00, -4.801512567875979e+05, -1.600356226068456e+01, -8.003082581322393e-01, -1.524471281708912e+01, -3.313273200680585e+05, -9.187976207619899e+01, -1.200106489585747e+05, -3.313273200680593e+05, -9.187976207619931e+01, -1.200106489585751e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05