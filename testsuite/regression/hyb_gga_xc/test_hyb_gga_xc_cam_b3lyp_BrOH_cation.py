
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_b3lyp_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.704876372212320e+01, -1.704878573204943e+01, -1.704893931332538e+01, -1.704861070134704e+01, -1.704877496474857e+01, -1.704877496474857e+01, -2.802864054116008e+00, -2.802835586216585e+00, -2.802246977555643e+00, -2.803809270942555e+00, -2.802898515098708e+00, -2.802898515098708e+00, -5.192742457360976e-01, -5.190452595341200e-01, -5.141949348209230e-01, -5.184341801891909e-01, -5.174442824177731e-01, -5.174442824177731e-01, -1.103503839598375e-01, -1.123009490560427e-01, -6.102458585636250e-01, -7.916277164931557e-02, -1.027013611779841e-01, -1.027013611779840e-01, -2.428557290817055e-02, -2.435598172542086e-02, -4.253782629539259e-02, -2.180341469377199e-02, -2.195457763329521e-02, -2.195457763329520e-02, -4.081209852046738e+00, -4.080704281423309e+00, -4.081195387606041e+00, -4.080748935000385e+00, -4.080948804581842e+00, -4.080948804581842e+00, -1.672306529541919e+00, -1.681009904937421e+00, -1.671918896793726e+00, -1.679613061767314e+00, -1.677308124646706e+00, -1.677308124646706e+00, -4.329797979837144e-01, -4.583906733425229e-01, -3.995356102695514e-01, -4.036045268764668e-01, -4.392774497553602e-01, -4.392774497553602e-01, -5.246489437311979e-02, -1.098717061706136e-01, -4.966470514319325e-02, -1.452342016035530e+00, -6.329442920217300e-02, -6.329442920217300e-02, -1.958497878507713e-02, -2.101845752742267e-02, -1.385439379485416e-02, -4.264236447751273e-02, -1.676412724956232e-02, -1.676412724956232e-02, -4.168597202633852e-01, -4.181261840102780e-01, -4.177220680945677e-01, -4.173518356143232e-01, -4.175402975733583e-01, -4.175402975733583e-01, -4.033585036016316e-01, -3.664781903559930e-01, -3.781836470141031e-01, -3.887421931639149e-01, -3.833731581055690e-01, -3.833731581055690e-01, -4.831653329408941e-01, -1.474855027548858e-01, -1.834285747608806e-01, -2.463906607726474e-01, -2.132551556418543e-01, -2.132551556418543e-01, -3.338531669693798e-01, -4.285011320690999e-02, -4.244804785978298e-02, -2.325371522105701e-01, -4.661260552653593e-02, -4.661260552653592e-02, -2.767783725646965e-02, -1.193183400129555e-02, -1.576135458530291e-02, -4.516129900465574e-02, -1.576825438144127e-02, -1.576825438144127e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_b3lyp_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.043875185812842e+01, -2.043872760727941e+01, -2.043884028119760e+01, -2.043879211471512e+01, -2.043910729909996e+01, -2.043917760959248e+01, -2.043810543645294e+01, -2.043790475382572e+01, -2.043880272647624e+01, -2.043840325053490e+01, -2.043880272647624e+01, -2.043840325053490e+01, -3.296968506346597e+00, -3.297044649394751e+00, -3.296996616047271e+00, -3.297074326437283e+00, -3.297695760167264e+00, -3.297873663111387e+00, -3.297021867537444e+00, -3.297194989008121e+00, -3.296415822872021e+00, -3.297825212915759e+00, -3.296415822872021e+00, -3.297825212915759e+00, -6.227400306838361e-01, -6.242204777538231e-01, -6.218297181384428e-01, -6.236241036211159e-01, -6.062956585324056e-01, -6.046215730520444e-01, -6.102439482211733e-01, -6.107692510726014e-01, -6.200031744754629e-01, -6.046165004724718e-01, -6.200031744754629e-01, -6.046165004724718e-01, -1.518591137666968e-01, -1.523015925122462e-01, -1.533239718389785e-01, -1.540084020945945e-01, -7.209392538507863e-01, -7.391434703646426e-01, -1.208462303980951e-01, -1.211975988614242e-01, -1.249198468794665e-01, -1.359849078874785e-01, -1.249198468794665e-01, -1.359849078874784e-01, -8.429915356063159e-03, -8.243525420278409e-03, -8.673866714776855e-03, -8.443205081116416e-03, -2.703810520081738e-02, -2.688992131787029e-02, -6.114331841437934e-03, -6.159463361531680e-03, -6.529462693785060e-03, -6.846163584867916e-03, -6.529462693785048e-03, -6.846163584867912e-03, -5.031431156728603e+00, -5.030232833569355e+00, -5.033433399004105e+00, -5.032170640747570e+00, -5.031541549078117e+00, -5.030300769147762e+00, -5.033262232440226e+00, -5.032058384967803e+00, -5.032455705929153e+00, -5.031206463967765e+00, -5.032455705929153e+00, -5.031206463967765e+00, -1.808927108739580e+00, -1.808848127249138e+00, -1.823137794921119e+00, -1.822704458010710e+00, -1.795801594065269e+00, -1.798574034466194e+00, -1.808035916443700e+00, -1.810976533749010e+00, -1.826125781013756e+00, -1.818682968370266e+00, -1.826125781013756e+00, -1.818682968370266e+00, -5.474781620755663e-01, -5.463947778584934e-01, -6.060327668423111e-01, -6.063792692289800e-01, -4.967117060665742e-01, -5.100327011450623e-01, -5.247018181956346e-01, -5.379801587359705e-01, -5.675446651853208e-01, -5.458910763686022e-01, -5.675446651853207e-01, -5.458910763686022e-01, -8.939225770997823e-02, -8.909400268719030e-02, -1.646595062646883e-01, -1.647455707756399e-01, -8.098802394026496e-02, -8.159851336061585e-02, -1.879455656046511e+00, -1.878716834560456e+00, -1.007635426589218e-01, -9.674939621184933e-02, -1.007635426589218e-01, -9.674939621184933e-02, -5.155820721266491e-03, -5.199980156537624e-03, -5.938707370094515e-03, -5.967765889659339e-03, -4.273092823602779e-03, -4.152602799825031e-03, -4.807903100651006e-02, -4.840932312569222e-02, -5.309082822206208e-03, -4.881552173640735e-03, -5.309082822206219e-03, -4.881552173640735e-03, -5.613153463304970e-01, -5.630920539433595e-01, -5.543080515246445e-01, -5.560752835704326e-01, -5.566521732950239e-01, -5.584353180131908e-01, -5.586923918948529e-01, -5.604554579249217e-01, -5.576610862027724e-01, -5.594340730071206e-01, -5.576610862027724e-01, -5.594340730071206e-01, -5.465500112394697e-01, -5.479875729805321e-01, -4.524566385855842e-01, -4.537063046389527e-01, -4.757878671405827e-01, -4.772913932616877e-01, -5.016307704191069e-01, -5.029799792746373e-01, -4.881731523230090e-01, -4.895173239306025e-01, -4.881731523230090e-01, -4.895173239306025e-01, -6.359186473224181e-01, -6.369544391424562e-01, -2.048939526822807e-01, -2.050078321735819e-01, -2.403361278173529e-01, -2.406325933728673e-01, -3.125417403678296e-01, -3.134978031509200e-01, -2.725166745294718e-01, -2.723069911021211e-01, -2.725166745294718e-01, -2.723069911021211e-01, -4.145388272245174e-01, -4.162527140140856e-01, -2.601589136543274e-02, -2.599031203656210e-02, -3.671781923873067e-02, -3.693275747133673e-02, -2.984870984096520e-01, -3.010936310679659e-01, -6.448947228380428e-02, -6.305482268657532e-02, -6.448947228380419e-02, -6.305482268657533e-02, -1.061334965416098e-02, -1.042625166039487e-02, -2.650281101107474e-03, -2.446996953964209e-03, -4.014414821305671e-03, -3.938618679791931e-03, -5.914235787999672e-02, -5.942119250676420e-02, -4.712893223667575e-03, -4.675485721670865e-03, -4.712893223667578e-03, -4.675485721670858e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_b3lyp_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.164496767220486e-08, 1.185558425401183e-11, -1.164503353323958e-08, -1.164490305683113e-08, 1.185541524986055e-11, -1.164498609179201e-08, -1.164455058503349e-08, 1.185461176244829e-11, -1.164452450785320e-08, -1.164528989852564e-08, 1.185712748825859e-11, -1.164545895067069e-08, -1.164493961075544e-08, 1.185581677785285e-11, -1.164497709080617e-08, -1.164493961075544e-08, 1.185581677785285e-11, -1.164497709080617e-08, -1.453580826606386e-05, 1.541225587214213e-07, -1.454131183920887e-05, -1.453623282939980e-05, 1.541207867069750e-07, -1.454214340964072e-05, -1.455055245733593e-05, 1.540597121148743e-07, -1.455409160610820e-05, -1.451640834243039e-05, 1.540100170374724e-07, -1.452092242732855e-05, -1.454262118078882e-05, 1.541018002171290e-07, -1.453324931497408e-05, -1.454262118078882e-05, 1.541018002171290e-07, -1.453324931497408e-05, -6.632558344086314e-03, 9.270722799357577e-04, -6.700979955351635e-03, -6.621729778887691e-03, 9.324368204854546e-04, -6.705659863494954e-03, -6.596106172932034e-03, 1.069807695322527e-03, -6.475380312310414e-03, -6.361084677018268e-03, 1.024962868820697e-03, -6.401173631183440e-03, -6.990054433081277e-03, 1.115015704194649e-03, -5.733710705527504e-03, -6.990054433081277e-03, 1.115015704194649e-03, -5.733710705527504e-03, -2.552257005714166e-01, 4.870430005817489e-01, -2.946236230678240e-01, -2.592056014164315e-01, 4.601020753985704e-01, -2.995807276178182e-01, -3.977672517581146e-03, 4.311409659161154e-04, -3.934922812718066e-03, -2.666881379439562e-01, 1.357480112752327e+00, -2.940182933162221e-01, -5.674483145126483e-01, 1.929431429665579e+00, -1.045795488011075e+00, -5.674483145126478e-01, 1.929431429665580e+00, -1.045795488011079e+00, -7.783771781602213e+03, 9.006507517295055e-08, -6.520342499831160e+03, -6.907595073044051e+03, 3.695673560289628e-07, -5.662099791855449e+03, -3.628474517991697e+01, 1.489288214502471e+01, -3.108045396151587e+01, -3.281511271821240e+04, 3.350885389426989e-17, -3.417850991657130e+04, -1.079860427668115e+04, 8.673770695270562e-13, -7.098237288691659e+04, -1.079860427668116e+04, 8.673770695270562e-13, -7.098237288691655e+04, -3.398097502712822e-06, 1.809299517292379e-08, -3.400964602330421e-06, -3.400652276724365e-06, 1.807545599629114e-08, -3.403432588439131e-06, -3.398204697229145e-06, 1.809212347410638e-08, -3.401022320598398e-06, -3.400394258280161e-06, 1.807663525628042e-08, -3.403267536419903e-06, -3.399433249024139e-06, 1.808413833346515e-08, -3.402207574164837e-06, -3.399433249024139e-06, 1.808413833346515e-08, -3.402207574164837e-06, -1.036289315505581e-04, 3.324691278596461e-06, -1.036432124390730e-04, -1.016277605624051e-04, 3.198712461362903e-06, -1.016954180266295e-04, -1.036220701494452e-04, 3.420793545345939e-06, -1.037687079873640e-04, -1.018921853082971e-04, 3.305373228471352e-06, -1.019960756054311e-04, -1.025561910057634e-04, 3.212781934568574e-06, -1.024734186898889e-04, -1.025561910057634e-04, 3.212781934568574e-06, -1.024734186898889e-04, -1.328542498644765e-02, 1.825453754003035e-03, -1.338335840403039e-02, -1.262415757065222e-02, 1.201192443406202e-03, -1.265481812624555e-02, -1.753521955592332e-02, 2.748455987594796e-03, -1.652713370358144e-02, -1.979224838133872e-02, 2.257366444272562e-03, -1.804513263593029e-02, -1.227107476101073e-02, 1.758833807333089e-03, -1.381747596710346e-02, -1.227107476101073e-02, 1.758833807333089e-03, -1.381747596710346e-02, -1.947715931893984e-01, 4.345108400965211e+00, -2.695031820362110e-01, -1.399850269145101e-01, 4.801730708975269e-01, -1.458424351031130e-01, -2.600926689321210e-01, 5.403210563745951e+00, -5.446372720855635e-01, -1.941928455114465e-04, 3.686127133635989e-06, -1.945338268816141e-04, -1.950342474132999e-01, 2.723523603638179e+00, -5.990333678631057e-01, -1.950342474132999e-01, 2.723523603638179e+00, -5.990333678631057e-01, -8.261026130570417e+04, 4.391412325733211e-24, -7.101092870428078e+04, -3.866851883428278e+04, 9.952946977518016e-18, -3.621287783575981e+04, -1.453821175779153e+05, 4.581716902253058e-25, -1.270080938879117e+05, -4.686233526077871e+00, 1.218782134268272e+01, -4.508716724144335e+00, -1.124845148056413e+05, 7.784361458058929e-20, -4.491904280703430e+04, -1.124845148056412e+05, 7.784361458058927e-20, -4.491904280703429e+04, -1.863079918610887e-02, 1.781738515145262e-03, -1.846038596355737e-02, -1.734518035306671e-02, 1.834071765417697e-03, -1.720085172232908e-02, -1.772932418754016e-02, 1.815455511527695e-03, -1.757973804653428e-02, -1.809906627763920e-02, 1.800280877020669e-03, -1.793610217690973e-02, -1.790805465673553e-02, 1.807857598825492e-03, -1.775185453325312e-02, -1.790805465673553e-02, 1.807857598825492e-03, -1.775185453325312e-02, -2.134735959972687e-02, 2.045475405717882e-03, -2.115259154133670e-02, -2.036950598463539e-02, 4.372269458584065e-03, -2.029687587555892e-02, -2.000268471869797e-02, 3.481409492639727e-03, -1.991279750743387e-02, -1.997915004935023e-02, 2.801304754186088e-03, -1.984203309891161e-02, -1.999005577697519e-02, 3.126795900343610e-03, -1.985503004542061e-02, -1.999005577697519e-02, 3.126795900343610e-03, -1.985503004542061e-02, -1.049333415260813e-02, 9.448281971276932e-04, -1.050250987433298e-02, -1.094889286500179e-01, 1.841168313207797e-01, -1.125781006273689e-01, -9.093306645488598e-02, 8.322264776862785e-02, -9.367124677969796e-02, -6.352430481421838e-02, 2.335777938713061e-02, -6.331312691547629e-02, -7.903020274620388e-02, 4.455605659310826e-02, -7.943964313662877e-02, -7.903020274620387e-02, 4.455605659310827e-02, -7.943964313662885e-02, -2.692087095858107e-02, 6.561578635696272e-03, -2.683824676672201e-02, -3.821903176062875e+01, 1.430892164753827e+01, -3.748844191221635e+01, -1.134712182678531e+01, 1.521830626896655e+01, -1.055639480023528e+01, -8.285426437703616e-02, 2.777420884472484e-02, -8.199846659247507e-02, -1.499341411555486e+00, 7.858164809404253e+00, -2.019232123023339e+00, -1.499341411555486e+00, 7.858164809404253e+00, -2.019232123023349e+00, -2.611921304629797e+03, 2.768157770177566e-04, -2.372781236228147e+03, -2.442203221496974e+06, 2.308592770889069e-85, -2.841809639174686e+06, -2.740398913973200e+05, 1.346473245321597e-36, -2.334059213225000e+05, -2.309388061861812e+00, 8.933405596013928e+00, -2.283500240519958e+00, -1.610395598220624e+05, 6.624839855865914e-22, -5.831617803583125e+04, -1.610395598220628e+05, 6.624839855865567e-22, -5.831617803583145e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05