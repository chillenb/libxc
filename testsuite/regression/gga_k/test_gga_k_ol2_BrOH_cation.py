
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ol2_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.189497569684302e+03, 2.189506608293508e+03, 2.189555652883530e+03, 2.189420995231288e+03, 2.189490292730562e+03, 2.189490292730562e+03, 5.912066587500613e+01, 5.912012224260351e+01, 5.911046040671592e+01, 5.914972870370143e+01, 5.912287458405148e+01, 5.912287458405148e+01, 2.298536747853849e+00, 2.296525755598626e+00, 2.258350977629433e+00, 2.291423891251667e+00, 2.300583649228987e+00, 2.300583649228987e+00, 2.054067606990658e-01, 2.070701555199457e-01, 3.093323505384034e+00, 1.641196068618667e-01, 2.051883947898220e-01, 2.051883947898220e-01, 8.123419561141375e-02, 7.904092913364695e-02, 1.076428276041002e-01, 8.517866498437060e-02, 7.636797957032587e-02, 7.636797957032576e-02, 1.283618189958650e+02, 1.283718381547417e+02, 1.283626330282944e+02, 1.283714718274643e+02, 1.283667005818590e+02, 1.283667005818590e+02, 2.051448478709818e+01, 2.074257736714970e+01, 2.045962090486582e+01, 2.065962355743879e+01, 2.066885667338364e+01, 2.066885667338364e+01, 1.667415009135546e+00, 1.884707110436362e+00, 1.441405157033050e+00, 1.488487499186973e+00, 1.716355023015357e+00, 1.716355023015357e+00, 1.405105369007703e-01, 2.482457605207332e-01, 1.326547408309552e-01, 1.707871768557743e+01, 1.352028683145466e-01, 1.352028683145466e-01, 7.423157551873627e-02, 7.855662508322847e-02, 3.133874505855969e-02, 1.048255421346884e-01, 4.602672641179349e-02, 4.602672641179354e-02, 1.594571773818470e+00, 1.592845490708704e+00, 1.593560867572540e+00, 1.594066193204105e+00, 1.593819617576263e+00, 1.593819617576263e+00, 1.504127098715010e+00, 1.234733974020177e+00, 1.305161564832714e+00, 1.379658204459880e+00, 1.340442373068314e+00, 1.340442373068314e+00, 2.074806408097290e+00, 3.376615172414740e-01, 4.257108359597211e-01, 6.263591336127960e-01, 5.091150710938286e-01, 5.091150710938286e-01, 1.049556839022507e+00, 1.124219912755648e-01, 1.142380433506492e-01, 5.622792187077705e-01, 1.035084689261657e-01, 1.035084689261657e-01, 9.055745670204438e-02, 3.562536968703393e-02, 5.094280148102760e-02, 1.017025340165582e-01, 4.127006501324907e-02, 4.127006501324902e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ol2_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.364014107252043e+03, 3.364005731390095e+03, 3.364039452437174e+03, 3.364024141679743e+03, 3.364120000897813e+03, 3.364139711836177e+03, 3.363833785034833e+03, 3.363774901111927e+03, 3.364029250727703e+03, 3.363920330664919e+03, 3.364029250727703e+03, 3.363920330664919e+03, 8.809910143470181e+01, 8.810172379670595e+01, 8.810028189571108e+01, 8.810288261704495e+01, 8.812869218335825e+01, 8.813716388967919e+01, 8.810786891025785e+01, 8.811588004499771e+01, 8.806765920591239e+01, 8.814445373697276e+01, 8.806765920591239e+01, 8.814445373697276e+01, 3.160144815987416e+00, 3.198488863077889e+00, 3.145749162781171e+00, 3.192788291167118e+00, 2.954108811120577e+00, 2.888505666258104e+00, 2.963960630195246e+00, 2.985223340055330e+00, 3.267128955508568e+00, 2.619091607556946e+00, 3.267128955508568e+00, 2.619091607556946e+00, 1.502979961516139e-01, 1.697501266364754e-01, 1.582905074640531e-01, 1.808119155527606e-01, 4.179357404913771e+00, 4.532542758939919e+00, 4.504914034117023e-02, 4.790014196850259e-02, 1.659748765217261e-01, -1.838709860558024e-02, 1.659748765217259e-01, -1.838709860558019e-02, -8.059205520194841e-02, -8.095377021993919e-02, -7.846838614401226e-02, -7.861179671840285e-02, -9.357912510088448e-02, -9.187179911489671e-02, -8.445753241896886e-02, -8.563393855393144e-02, -8.073428878524659e-02, -5.077243991460548e-02, -8.073428878524645e-02, -5.077243991460544e-02, 2.016279469489382e+02, 2.015298748466805e+02, 2.017399039511560e+02, 2.016380066256679e+02, 2.016344083201120e+02, 2.015337631660123e+02, 2.017304727050490e+02, 2.016319948018489e+02, 2.016851158401216e+02, 2.015841623158821e+02, 2.016851158401216e+02, 2.015841623158821e+02, 2.612477725594761e+01, 2.612194569829011e+01, 2.658212986476451e+01, 2.656599184349845e+01, 2.561330222067983e+01, 2.575890653628612e+01, 2.600780411913313e+01, 2.615733581389330e+01, 2.675933192504338e+01, 2.640743719057589e+01, 2.675933192504338e+01, 2.640743719057589e+01, 2.529793310520130e+00, 2.516684841235747e+00, 3.056262413827158e+00, 3.059876704542292e+00, 2.066393249150599e+00, 2.229852741455192e+00, 2.310774597426466e+00, 2.465969054997492e+00, 2.749541265421618e+00, 2.478540260275652e+00, 2.749541265421618e+00, 2.478540260275653e+00, -4.035786767793346e-02, -3.765853381551559e-02, 1.170965210497958e-01, 1.193277589904449e-01, -4.874394826258889e-02, -4.268954562529653e-02, 2.776463813684806e+01, 2.774008655807045e+01, -7.967543987401680e-03, 1.621674526272093e-02, -7.967543987401680e-03, 1.621674526272093e-02, -7.131969236241058e-02, -7.665932127126369e-02, -7.681558082398332e-02, -7.994478552516451e-02, -3.216839339153848e-02, -3.048761428994985e-02, -6.479890698928642e-02, -6.623062454024221e-02, -4.564275843273538e-02, -4.599072098834920e-02, -4.564275843273544e-02, -4.599072098834926e-02, 2.621176093477283e+00, 2.641676356290596e+00, 2.577362348904016e+00, 2.597981918143620e+00, 2.592968183566924e+00, 2.613650361325442e+00, 2.605841739586882e+00, 2.626303858775363e+00, 2.599424173383989e+00, 2.619992371768182e+00, 2.599424173383989e+00, 2.619992371768182e+00, 2.485119042551491e+00, 2.501589224180281e+00, 1.708944459327880e+00, 1.725508508231555e+00, 1.922014592628814e+00, 1.940787048998304e+00, 2.143976735014101e+00, 2.159563902537836e+00, 2.030989547735092e+00, 2.046691597936541e+00, 2.030989547735092e+00, 2.046691597936541e+00, 3.351777856194028e+00, 3.364151919164525e+00, 2.450292998125448e-01, 2.495290208996865e-01, 4.118132914247568e-01, 4.239643840354403e-01, 8.400838252254098e-01, 8.502472372861293e-01, 6.036924889462826e-01, 6.040442194864349e-01, 6.036924889462824e-01, 6.040442194864348e-01, 1.441116245706573e+00, 1.463878761219543e+00, -9.885701773144816e-02, -9.862982463796205e-02, -9.016011687017543e-02, -8.752799034434185e-02, 7.921079712539715e-01, 8.182942657527371e-01, -4.679692663795267e-02, -3.340011143732802e-02, -4.679692663795265e-02, -3.340011143732800e-02, -9.019495188375445e-02, -8.920594880053313e-02, -4.069263715796933e-02, -3.057258577083595e-02, -5.177218062997049e-02, -5.017169044941823e-02, -4.578936731464150e-02, -4.521407708875787e-02, -3.177339467573509e-02, -4.515722585885137e-02, -3.177339467573501e-02, -4.515722585885135e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ol2_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.443919951448076e-06, 0.000000000000000e+00, 1.443925557473018e-06, 1.443906307320328e-06, 0.000000000000000e+00, 1.443915645195578e-06, 1.443859427791230e-06, 0.000000000000000e+00, 1.443849374859741e-06, 1.444013601677406e-06, 0.000000000000000e+00, 1.444045859025384e-06, 1.443911976917346e-06, 0.000000000000000e+00, 1.443964937106235e-06, 1.443911976917346e-06, 0.000000000000000e+00, 1.443964937106235e-06, 3.339667097158492e-04, 0.000000000000000e+00, 3.339823221226879e-04, 3.339638202902056e-04, 0.000000000000000e+00, 3.339814968294893e-04, 3.339145871883982e-04, 0.000000000000000e+00, 3.338958577162006e-04, 3.338382703253982e-04, 0.000000000000000e+00, 3.338263361927705e-04, 3.341335350260239e-04, 0.000000000000000e+00, 3.337629500802371e-04, 3.341335350260239e-04, 0.000000000000000e+00, 3.337629500802371e-04, 4.674065158079605e-02, 0.000000000000000e+00, 4.619456650952047e-02, 4.695782339191657e-02, 0.000000000000000e+00, 4.628473867358482e-02, 4.999821332452930e-02, 0.000000000000000e+00, 5.104035412805608e-02, 4.943702796335992e-02, 0.000000000000000e+00, 4.911893187396441e-02, 4.515674946760165e-02, 0.000000000000000e+00, 5.488704342369977e-02, 4.515674946760165e-02, 0.000000000000000e+00, 5.488704342369977e-02, 2.858998386916008e+00, 0.000000000000000e+00, 2.549802832134759e+00, 2.755907175809216e+00, 0.000000000000000e+00, 2.419480676850598e+00, 3.083595053042211e-02, 0.000000000000000e+00, 2.793957810498530e-02, 6.288987183524677e+00, 0.000000000000000e+00, 6.035648046056383e+00, 2.511268440175330e+00, 0.000000000000000e+00, 1.626155075646606e+01, 2.511268440175330e+00, 0.000000000000000e+00, 1.626155075646605e+01, 7.027225610781662e+04, 0.000000000000000e+00, 5.856688632333623e+04, 6.107291960348601e+04, 0.000000000000000e+00, 4.967907525163416e+04, 3.552402659907256e+02, 0.000000000000000e+00, 3.016563220560201e+02, 3.235066884589649e+05, 0.000000000000000e+00, 3.402326911896940e+05, 9.898051361973649e+04, 0.000000000000000e+00, 5.371167953706480e+05, 9.898051361973649e+04, 0.000000000000000e+00, 5.371167953706476e+05, 9.993876193765883e-05, 0.000000000000000e+00, 1.000102502956849e-04, 9.989181835234978e-05, 0.000000000000000e+00, 9.996485720211016e-05, 9.993581049968539e-05, 0.000000000000000e+00, 1.000084451506334e-04, 9.989553114621000e-05, 0.000000000000000e+00, 9.996720860316338e-05, 9.991495633592643e-05, 0.000000000000000e+00, 9.998747389351079e-05, 9.991495633592643e-05, 0.000000000000000e+00, 9.998747389351079e-05, 1.869385420684771e-03, 0.000000000000000e+00, 1.869668689582476e-03, 1.828736168142753e-03, 0.000000000000000e+00, 1.830228687306984e-03, 1.904340862326385e-03, 0.000000000000000e+00, 1.894533172627507e-03, 1.868240996758753e-03, 0.000000000000000e+00, 1.858217873091353e-03, 1.821308441955463e-03, 0.000000000000000e+00, 1.845350217366976e-03, 1.821308441955463e-03, 0.000000000000000e+00, 1.845350217366976e-03, 6.931281837415693e-02, 0.000000000000000e+00, 6.986736580464466e-02, 5.540749140476843e-02, 0.000000000000000e+00, 5.538613188837467e-02, 9.278242447648623e-02, 0.000000000000000e+00, 8.329275840793321e-02, 8.323688718659610e-02, 0.000000000000000e+00, 7.541541683417873e-02, 6.166454188868781e-02, 0.000000000000000e+00, 7.186387486104528e-02, 6.166454188868783e-02, 0.000000000000000e+00, 7.186387486104528e-02, 1.867051810004128e+01, 0.000000000000000e+00, 1.814642555929030e+01, 2.703254046574244e+00, 0.000000000000000e+00, 2.663185668782889e+00, 2.522069659981235e+01, 0.000000000000000e+00, 2.155605502000337e+01, 2.030586093897221e-03, 0.000000000000000e+00, 2.033442769725570e-03, 1.246293231161667e+01, 0.000000000000000e+00, 1.048395543000859e+01, 1.246293231161667e+01, 0.000000000000000e+00, 1.048395543000859e+01, 7.662495723833998e+05, 0.000000000000000e+00, 6.826841577843475e+05, 3.631779460186475e+05, 0.000000000000000e+00, 3.471746550498408e+05, 8.685799853285652e+05, 0.000000000000000e+00, 7.312712629357064e+05, 7.710794051851634e+01, 0.000000000000000e+00, 7.570291041502864e+01, 8.152459908707459e+05, 0.000000000000000e+00, 3.147553663466809e+05, 8.152459908707457e+05, 0.000000000000000e+00, 3.147553663466810e+05, 7.577902519307878e-02, 0.000000000000000e+00, 7.518519174918402e-02, 7.173105355417295e-02, 0.000000000000000e+00, 7.094693473523471e-02, 7.205327080238620e-02, 0.000000000000000e+00, 7.129775136715782e-02, 7.285429344377863e-02, 0.000000000000000e+00, 7.212269857093509e-02, 7.236334799497955e-02, 0.000000000000000e+00, 7.161900556520065e-02, 7.236334799497955e-02, 0.000000000000000e+00, 7.161900556520065e-02, 9.483814782252209e-02, 0.000000000000000e+00, 9.412833725312818e-02, 1.182309149926516e-01, 0.000000000000000e+00, 1.167110102469162e-01, 1.031150735627866e-01, 0.000000000000000e+00, 1.017619981532648e-01, 9.066350019601334e-02, 0.000000000000000e+00, 8.972132192355631e-02, 9.666409886769614e-02, 0.000000000000000e+00, 9.559943944168448e-02, 9.666409886769614e-02, 0.000000000000000e+00, 9.559943944168448e-02, 4.802581498064753e-02, 0.000000000000000e+00, 4.785000862712532e-02, 1.342078308745502e+00, 0.000000000000000e+00, 1.319037705886939e+00, 7.799009152684193e-01, 0.000000000000000e+00, 7.569099337585532e-01, 3.364503828560764e-01, 0.000000000000000e+00, 3.309372748608005e-01, 5.067545436873093e-01, 0.000000000000000e+00, 5.072950043126567e-01, 5.067545436873098e-01, 0.000000000000000e+00, 5.072950043126568e-01, 1.520890055468100e-01, 0.000000000000000e+00, 1.489980392582148e-01, 3.759445189335026e+02, 0.000000000000000e+00, 3.683320122771724e+02, 1.544639024568736e+02, 0.000000000000000e+00, 1.399353176245565e+02, 3.812740053543655e-01, 0.000000000000000e+00, 3.645781596620689e-01, 4.028600071462046e+01, 0.000000000000000e+00, 3.422100422460577e+01, 4.028600071462046e+01, 0.000000000000000e+00, 3.422100422460579e+01, 2.400891850358103e+04, 0.000000000000000e+00, 2.157612390762197e+04, 1.865027159492778e+07, 0.000000000000000e+00, 1.852297451356899e+07, 2.213808764755875e+06, 0.000000000000000e+00, 1.840046209765836e+06, 4.580557954128414e+01, 0.000000000000000e+00, 4.362989450211312e+01, 9.592826181599102e+05, 0.000000000000000e+00, 4.087624143447934e+05, 9.592826181599123e+05, 0.000000000000000e+00, 4.087624143447946e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05