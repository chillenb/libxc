
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th_fcfo_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.985807410551473e+01, -2.985713285148416e+01, -2.985493779573371e+01, -2.986889923649666e+01, -2.986129864526667e+01, -2.986129864526667e+01, -3.815719015225997e+00, -3.815742343892528e+00, -3.816374526864371e+00, -3.815966771963124e+00, -3.815833260110657e+00, -3.815833260110657e+00, -8.980340096857885e-01, -8.978349939398397e-01, -8.959899162706985e-01, -9.004860195652763e-01, -9.005362318073094e-01, -9.005362318073094e-01, -4.397622199035824e-01, -4.395796179527188e-01, -1.002612882770876e+00, -4.229029656696516e-01, -4.315911607678860e-01, -4.315911607678860e-01, -2.041612413493332e-01, -2.081576794977969e-01, -3.473366595294093e-01, -1.458007910451671e-01, -2.201142582782851e-01, -2.201142582782850e-01, -5.620216386562551e+00, -5.621962079230092e+00, -5.620296402074533e+00, -5.621836835211377e+00, -5.621102000957981e+00, -5.621102000957981e+00, -2.248424979803854e+00, -2.259924831691968e+00, -2.245923106290186e+00, -2.255792817797542e+00, -2.256231331461426e+00, -2.256231331461426e+00, -7.980612653505067e-01, -8.402061246878033e-01, -7.581311906840290e-01, -7.675009179295404e-01, -8.064827864946137e-01, -8.064827864946136e-01, -4.080512728841638e-01, -4.737653543309268e-01, -4.001135655216082e-01, -2.215356717297511e+00, -4.015179419514525e-01, -4.015179419514525e-01, -1.199959323176619e-01, -1.453939735103989e-01, -1.275931588045440e-01, -3.644851377471169e-01, -1.549126368962423e-01, -1.549126368962423e-01, -7.922167937028965e-01, -7.883188702405436e-01, -7.893900810187486e-01, -7.904744218311569e-01, -7.899039217750433e-01, -7.899039217750433e-01, -7.764127883364572e-01, -7.223291766507322e-01, -7.327900810662077e-01, -7.458575125735946e-01, -7.386626555214582e-01, -7.386626555214582e-01, -8.713916828693465e-01, -5.101366029900825e-01, -5.356262090986580e-01, -5.825980268255441e-01, -5.550662703356340e-01, -5.550662703356338e-01, -6.846839887401026e-01, -3.496295089714309e-01, -3.658966384991351e-01, -5.592150257514580e-01, -3.688133175809780e-01, -3.688133175809779e-01, -2.379920357567218e-01, -4.309853920768487e-02, -9.776064337029813e-02, -3.659263213306712e-01, -1.502456328455259e-01, -1.502456328455258e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th_fcfo_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [4.726749540729444e+01, 4.726459345040706e+01, 4.726930562689752e+01, 4.726441212039033e+01, 4.726766368141963e+01, 4.727307403416966e+01, 4.726683548295838e+01, 4.724967115935964e+01, 4.727897687888292e+01, 4.725025506406234e+01, 4.727897687888292e+01, 4.725025506406234e+01, -5.229229505551168e+00, -5.229757701465426e+00, -5.229261399863513e+00, -5.229864546593000e+00, -5.231690401749318e+00, -5.230893303117877e+00, -5.229545607722602e+00, -5.229002094031098e+00, -5.236944108218577e+00, -5.222460938307544e+00, -5.236944108218577e+00, -5.222460938307544e+00, -1.003168973404249e+00, -1.006082762875302e+00, -1.000621770483886e+00, -1.004201731121704e+00, -9.480657180400662e-01, -9.429002375221088e-01, -9.527194483796257e-01, -9.543368297935685e-01, -9.756613327040177e-01, -9.272084661104286e-01, -9.756613327040177e-01, -9.272084661104286e-01, -3.668715785203048e-01, -3.680471470217390e-01, -3.715705260452399e-01, -3.730036318374474e-01, -1.146102406113525e+00, -1.168752442490867e+00, -3.028977254286544e-01, -3.027251211210049e-01, -3.336826450044332e-01, -2.776802800429239e-01, -3.336826450044332e-01, -2.776802800429239e-01, -1.882155389198402e-01, -1.880766712239692e-01, -1.817481558796787e-01, -1.818421364468695e-01, -1.720026328980671e-01, -1.725972010243042e-01, -2.417681273967260e-01, -2.429721681838938e-01, -1.470673912115912e-01, -1.279010865885503e-01, -1.470673912115907e-01, -1.279010865885499e-01, -7.520009766768799e+00, -7.545747976971450e+00, -7.519259034889494e+00, -7.545688966698570e+00, -7.519761829187290e+00, -7.545979658012887e+00, -7.519596460599782e+00, -7.545430858134915e+00, -7.519561167078468e+00, -7.545777466942610e+00, -7.519561167078468e+00, -7.545777466942610e+00, -2.529716999730706e+00, -2.530067761960431e+00, -2.562748786563579e+00, -2.564686908810491e+00, -2.487543435665715e+00, -2.475768176593437e+00, -2.518110798839300e+00, -2.505560563576001e+00, -2.556024562108752e+00, -2.587199119825764e+00, -2.556024562108752e+00, -2.587199119825764e+00, -9.455387895936617e-01, -9.436595757876708e-01, -1.056297905223897e+00, -1.056648205586808e+00, -8.677943311283622e-01, -8.920081329649343e-01, -9.336080453150257e-01, -9.569815011114394e-01, -9.801013117944727e-01, -9.435947548463119e-01, -9.801013117944729e-01, -9.435947548463122e-01, -2.425035410367550e-01, -2.422984266552675e-01, -3.489519004871934e-01, -3.493185532248724e-01, -2.344685052365793e-01, -2.330932826658491e-01, -3.055964577430119e+00, -3.058781058303098e+00, -2.712946993610780e-01, -2.689998291850618e-01, -2.712946993610780e-01, -2.689998291850618e-01, -2.453054603425001e-01, -2.415542077189159e-01, -2.302455677045499e-01, -2.292502798678888e-01, -1.329441704071371e-01, -1.310517136184183e-01, -1.992488419302671e-01, -1.990333171815874e-01, -1.490723046427918e-01, -1.444568157284352e-01, -1.490723046427921e-01, -1.444568157284354e-01, -9.986159017472782e-01, -1.001596979533476e+00, -9.819313011576533e-01, -9.848312303288517e-01, -9.874592357045652e-01, -9.903892920156255e-01, -9.922661536341165e-01, -9.952002531376062e-01, -9.898311474345624e-01, -9.927622711679819e-01, -9.898311474345624e-01, -9.927622711679819e-01, -9.799005055931935e-01, -9.824508254460244e-01, -7.871946591245393e-01, -7.898575394530665e-01, -8.394417164862725e-01, -8.422867060642668e-01, -8.930470268775039e-01, -8.954208160393283e-01, -8.658358303332454e-01, -8.682751125527606e-01, -8.658358303332454e-01, -8.682751125527606e-01, -1.098727696501044e+00, -1.099992200781570e+00, -4.069491529020465e-01, -4.078483619122865e-01, -4.704727524481671e-01, -4.727609566376397e-01, -6.058680849422484e-01, -6.079391357730295e-01, -5.344493151140948e-01, -5.343455171714694e-01, -5.344493151140949e-01, -5.343455171714692e-01, -7.375848911887407e-01, -7.414415570960360e-01, -1.717269595161466e-01, -1.718743332975317e-01, -1.837620744249223e-01, -1.837413907375366e-01, -6.003897722182081e-01, -6.055635279332086e-01, -2.239440390731880e-01, -2.207879051834163e-01, -2.239440390731880e-01, -2.207879051834163e-01, -1.802272058051349e-01, -1.809810334987111e-01, -2.127187914643646e-01, -2.123695566893502e-01, -2.119875492618666e-01, -2.059326122851697e-01, -2.165196570163955e-01, -2.156133515740974e-01, -1.378984152762907e-01, -1.341930542311250e-01, -1.378984152762906e-01, -1.341930542311248e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th_fcfo_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.736003068189417e-05, -3.580441863056711e-05, 1.736002952655814e-05, 1.735994320123639e-05, -3.580424662636960e-05, 1.735994225479392e-05, 1.735953299481658e-05, -3.580342886658840e-05, 1.735953169158911e-05, 1.736083227960222e-05, -3.580598909641096e-05, 1.736083332800810e-05, 1.736014923431264e-05, -3.580465511280819e-05, 1.736015976042617e-05, 1.736014923431264e-05, -3.580465511280819e-05, 1.736015976042617e-05, 2.486513757242278e-04, -4.842847588976406e-04, 2.486667340713760e-04, 2.486525642770275e-04, -4.842838202043793e-04, 2.486689966463174e-04, 2.486891397539031e-04, -4.842514647404668e-04, 2.487010869854940e-04, 2.485844607642707e-04, -4.842251286729718e-04, 2.485987576490174e-04, 2.486566312948435e-04, -4.842733854799265e-04, 2.486546740933336e-04, 2.486566312948435e-04, -4.842733854799265e-04, 2.486546740933336e-04, -7.913652075015399e-03, -1.922101790907514e-04, -7.847225490860083e-03, -7.992519132272372e-03, -2.004209249797423e-04, -7.912769527851944e-03, -9.708991391430380e-03, -4.253682123475294e-04, -9.773850909734124e-03, -9.407932663952847e-03, -3.513672202671765e-04, -9.383803768870901e-03, -9.041259110379037e-03, -3.494741957713977e-04, -9.825987884529498e-03, -9.041259110379037e-03, -3.494741957713977e-04, -9.825987884529498e-03, -2.014951309833023e+00, -7.605322516394286e-01, -1.867726764580044e+00, -1.950700088541234e+00, -7.223875531217243e-01, -1.788477771945325e+00, -3.850558479823554e-03, 3.950787627834134e-04, -3.950673356039105e-03, -4.685975824101587e+00, -1.878026768257884e+00, -4.513415423125745e+00, -2.488060185484540e+00, -1.588735740310955e+00, -1.001056546631234e+01, -2.488060185484540e+00, -1.588735740310956e+00, -1.001056546631234e+01, -8.866198645296654e+04, 1.228780812998210e+05, -7.155057884238374e+04, -7.822497303439678e+04, 1.003313376809607e+05, -6.153482655254077e+04, -3.366335117626123e+02, -6.702683026837013e+00, -2.850567131995988e+02, -3.943183430768531e+05, 1.029022170881546e+06, -4.154707590198613e+05, -1.155201055459431e+05, 4.823605669970977e+05, -1.057145754931697e+06, -1.155201055459432e+05, 4.823605669970978e+05, -1.057145754931697e+06, 1.584469479924432e-04, -3.123298322299263e-04, 1.584473396527015e-04, 1.584108849139974e-04, -3.122585687969977e-04, 1.584112194360496e-04, 1.584451513652243e-04, -3.123262843315607e-04, 1.584455092643936e-04, 1.584132736453702e-04, -3.122633714435942e-04, 1.584136626970115e-04, 1.584287615874496e-04, -3.122938513436167e-04, 1.584291011758625e-04, 1.584287615874496e-04, -3.122938513436167e-04, 1.584291011758625e-04, 2.133531485679738e-04, -6.152682486594997e-04, 2.133847827648291e-04, 2.192377127383021e-04, -6.166107344357978e-04, 2.193854154883149e-04, 2.053409521063788e-04, -6.142122108428089e-04, 2.057326529911714e-04, 2.108659224655313e-04, -6.154932490187012e-04, 2.111768799061690e-04, 2.206634584224878e-04, -6.165489632046692e-04, 2.201809917634643e-04, 2.206634584224878e-04, -6.165489632046692e-04, 2.201809917634643e-04, -1.117931226016735e-02, -1.870651331654352e-03, -1.106636792728759e-02, 7.919000676500735e-04, -6.574779674526360e-04, 1.011071306907498e-03, -1.673268473059367e-02, -3.711578564475799e-03, -1.758055804977093e-02, -6.325102705157214e-03, -2.687120481851287e-03, -8.575323736977022e-03, -1.051942624438231e-02, -1.566619240602880e-03, -8.705510211767058e-03, -1.051942624438234e-02, -1.566619240602821e-03, -8.705510211767097e-03, -1.445378742642073e+01, -5.475752191406754e+00, -1.418232730206226e+01, -1.618136698029948e+00, -7.557653444832588e-01, -1.600681151125995e+00, -2.012238289137787e+01, -6.759185141893004e+00, -1.746998237462664e+01, 6.705396831931959e-04, -6.110667930178880e-04, 6.722833383983516e-04, -9.669707770301615e+00, -3.503628502409977e+00, -8.956853321940461e+00, -9.669707770301615e+00, -3.503628502409977e+00, -8.956853321940461e+00, -1.053439451618416e+06, 2.744110150144340e+06, -8.781827348861126e+05, -4.729871726092675e+05, 1.121601769994472e+06, -4.376954273644686e+05, -1.893496274899890e+06, 3.094080784450139e+06, -1.608291311858420e+06, -7.173426400156121e+01, -1.616109744723343e+01, -6.979492500936760e+01, -1.582709619540070e+06, 1.615024973327004e+06, -5.315356864268454e+05, -1.582709619540069e+06, 1.615024973327005e+06, -5.315356864268456e+05, 1.069670928850445e-02, -1.780631800273368e-03, 1.068019775772208e-02, -1.981383567360236e-03, -1.887776806531090e-03, -2.086583601765478e-03, 6.647824551189534e-04, -1.849573153921285e-03, 5.819662147869412e-04, 3.894762538594356e-03, -1.818517911548498e-03, 3.780096309524805e-03, 2.120468612850930e-03, -1.834016482228373e-03, 2.022964886030367e-03, 2.120468612850930e-03, -1.834016482228373e-03, 2.022964886030367e-03, 2.341915320548664e-02, -2.327736248875659e-03, 2.302443764937499e-02, -2.914245512078830e-02, -7.490745588628430e-03, -2.919211868180233e-02, -2.241272934572282e-02, -5.475254922580576e-03, -2.247486233574661e-02, -1.533723441867064e-02, -3.960505239993681e-03, -1.544155692413674e-02, -1.906502286697806e-02, -4.682077729614121e-03, -1.916073686338769e-02, -1.906502286697806e-02, -4.682077729614121e-03, -1.916073686338769e-02, 1.136139868802318e-03, -2.207307545342694e-04, 1.352693797088720e-03, -6.987336392286774e-01, -3.275645488458117e-01, -6.918649025975501e-01, -3.693456859306608e-01, -1.616007068153949e-01, -3.650094272535720e-01, -1.363232869450774e-01, -4.895950866458904e-02, -1.355140769128412e-01, -2.270977155133252e-01, -9.110208639259056e-02, -2.276855364631625e-01, -2.270977155133251e-01, -9.110208639259097e-02, -2.276855364631626e-01, -4.202710378070036e-02, -1.247298424653217e-02, -4.203570043123651e-02, -3.483526240931872e+02, 1.445616050998163e+00, -3.411687716364010e+02, -1.378454892809561e+02, -2.050004592344442e+01, -1.257485801520954e+02, -1.739617199164286e-01, -5.790784726849241e-02, -1.709772861456786e-01, -3.630189065565934e+01, -9.926260515327192e+00, -3.283068917649651e+01, -3.630189065565940e+01, -9.926260515327153e+00, -3.283068917649653e+01, -2.772324006923959e+04, 3.061072759629539e+04, -2.466828609794994e+04, -3.520779440504533e+07, 1.482566058766001e+08, -4.133835064912194e+07, -3.708885573075382e+06, 9.812114599742269e+06, -3.038422278510262e+06, -4.302627855677576e+01, -1.139804738920773e+01, -4.093593870835117e+01, -2.243098156765813e+06, 2.146831197521955e+06, -6.990829094264745e+05, -2.243098156765818e+06, 2.146831197521963e+06, -6.990829094264775e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05