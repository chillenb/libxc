
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_am05_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.508909236535069e-01, -1.508912301279497e-01, -1.508922497349555e-01, -1.508876972668355e-01, -1.508901313428568e-01, -1.508901313428568e-01, -9.985796459689118e-02, -9.985909436280641e-02, -9.988588502319050e-02, -9.984424484446991e-02, -9.985993028353617e-02, -9.985993028353617e-02, -5.880282897583706e-02, -5.874356250781938e-02, -5.734883839291801e-02, -5.755720010743295e-02, -5.753206185748719e-02, -5.753206185748719e-02, -3.148340728707918e-02, -3.175863999852867e-02, -6.241999132607277e-02, -2.703599494135868e-02, -2.298114861222919e-02, -2.298114861222920e-02, -3.056036428937849e-03, -3.188679407272854e-03, -1.193817583240610e-02, -1.907897082370560e-03, -1.982830162673062e-03, -1.982830162673062e-03, -1.140050990476124e-01, -1.140532464928599e-01, -1.140071263027909e-01, -1.140496315873309e-01, -1.140295843312797e-01, -1.140295843312797e-01, -8.165363459508118e-02, -8.190822853497240e-02, -8.134199851064845e-02, -8.156492033363964e-02, -8.194836774219422e-02, -8.194836774219422e-02, -5.797123535800747e-02, -6.328224805582294e-02, -5.586817058228744e-02, -5.979101485761746e-02, -5.854954719695549e-02, -5.854954719695546e-02, -2.198050827178925e-02, -3.124333969757494e-02, -2.096164320328592e-02, -9.076820083942470e-02, -2.406946639722585e-02, -2.406946639722585e-02, -1.515518510206795e-03, -1.870343504650077e-03, -1.471588990187622e-03, -1.648868396446204e-02, -1.638564396286842e-03, -1.638564396286842e-03, -6.296899158909597e-02, -6.145517293860304e-02, -6.194925369235334e-02, -6.238815272369840e-02, -6.216509353660326e-02, -6.216509353660326e-02, -6.280746019153140e-02, -5.247403872381655e-02, -5.460603740730068e-02, -5.723991116982019e-02, -5.583486612082512e-02, -5.583486612082512e-02, -6.418011748617841e-02, -3.519793368132586e-02, -3.869670196999871e-02, -4.528148288237812e-02, -4.177421783775520e-02, -4.177421783775520e-02, -5.074120989556452e-02, -1.160758170803306e-02, -1.430605466586490e-02, -4.534939444356316e-02, -1.912240483494045e-02, -1.912240483494045e-02, -4.079554761670768e-03, -5.638698148233334e-04, -1.113466166686707e-03, -1.843904006356023e-02, -1.546841694442717e-03, -1.546841694442719e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_am05_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.807331092595710e-01, -1.807334753254694e-01, -1.807332686137757e-01, -1.807338276133532e-01, -1.807346855258601e-01, -1.807342041738090e-01, -1.807297900704172e-01, -1.807314965362634e-01, -1.807314970125168e-01, -1.807338421590242e-01, -1.807314970125168e-01, -1.807338421590242e-01, -1.224484807353774e-01, -1.224585692379059e-01, -1.224493967207644e-01, -1.224603275901457e-01, -1.224848002503340e-01, -1.224881579884323e-01, -1.224343219573071e-01, -1.224399643498092e-01, -1.224829508234776e-01, -1.224287430985757e-01, -1.224829508234776e-01, -1.224287430985757e-01, -7.375792810474049e-02, -7.357374656534474e-02, -7.368344747227623e-02, -7.345667763324426e-02, -7.100782918108077e-02, -7.135921562031519e-02, -7.151566630723982e-02, -7.141345301481926e-02, -6.997198193949172e-02, -7.311641041335429e-02, -6.997198193949172e-02, -7.311641041335429e-02, -3.982342623938533e-02, -3.752566740045958e-02, -4.043272982372275e-02, -3.780332334851544e-02, -7.969350929860211e-02, -7.673483753886905e-02, -3.304751157432268e-02, -3.228028108663697e-02, -2.337570383303557e-02, -5.932314707443608e-02, -2.337570383303559e-02, -5.932314707443606e-02, -4.102554383061969e-03, -3.763051879225137e-03, -4.306057172067764e-03, -3.902967456317266e-03, -1.543583387945757e-02, -1.407907300653258e-02, -2.438868717647999e-03, -2.495340609501640e-03, -2.215525570313600e-03, -4.341466300741679e-03, -2.215525570313600e-03, -4.341466300741680e-03, -1.373609582017947e-01, -1.374114549263233e-01, -1.373806658673871e-01, -1.374321731321376e-01, -1.373614945705771e-01, -1.374126994243062e-01, -1.373796046128093e-01, -1.374302905946401e-01, -1.373709667706934e-01, -1.374220821285233e-01, -1.373709667706934e-01, -1.374220821285233e-01, -9.908551104236744e-02, -9.909293184078305e-02, -9.945755244797864e-02, -9.949592037841809e-02, -9.857627814922036e-02, -9.844897000338693e-02, -9.892410750714274e-02, -9.878314027077070e-02, -9.940119624229299e-02, -9.977290281900072e-02, -9.940119624229299e-02, -9.977290281900072e-02, -7.307029648200873e-02, -7.338683282461342e-02, -7.636381388746921e-02, -7.627087713321216e-02, -7.273720029971456e-02, -6.895743634153073e-02, -7.536721182287141e-02, -7.145235698595841e-02, -7.109390734628569e-02, -7.700636840680802e-02, -7.109390734628568e-02, -7.700636840680797e-02, -2.668821802133405e-02, -2.625608138311857e-02, -3.784435121702830e-02, -3.753170376657436e-02, -2.652346689872508e-02, -2.421148524052138e-02, -1.068746520607261e-01, -1.069501154168705e-01, -3.055154294761527e-02, -2.779504377270141e-02, -3.055154294761527e-02, -2.779504377270141e-02, -2.019354416495487e-03, -1.918064074914244e-03, -2.443872282577690e-03, -2.394446306192532e-03, -1.989029003996349e-03, -1.842647537443335e-03, -2.016643508523253e-02, -1.995009130097905e-02, -2.792695895102175e-03, -1.859240016398113e-03, -2.792695895102176e-03, -1.859240016398112e-03, -7.336526182968232e-02, -7.284453785962314e-02, -7.434327574164958e-02, -7.383814478280887e-02, -7.412540465175579e-02, -7.360945264623135e-02, -7.384515065022519e-02, -7.333305859655588e-02, -7.399708336525973e-02, -7.348279546300839e-02, -7.399708336525973e-02, -7.348279546300839e-02, -7.202947675815916e-02, -7.160680720627753e-02, -6.642507977644624e-02, -6.602746775054520e-02, -6.944524512840604e-02, -6.901943600202505e-02, -7.209846758013866e-02, -7.170332849374256e-02, -7.084890411411847e-02, -7.045046864658730e-02, -7.084890411411847e-02, -7.045046864658730e-02, -7.777145331762833e-02, -7.751630071835613e-02, -4.291764464139542e-02, -4.252583054287429e-02, -4.791256257565199e-02, -4.722553784200993e-02, -5.760917918037509e-02, -5.714382935781639e-02, -5.213256583659420e-02, -5.219719826264978e-02, -5.213256583659422e-02, -5.219719826264978e-02, -6.441552381123021e-02, -6.382214148226403e-02, -1.439451540016091e-02, -1.423003656335585e-02, -1.800942477561652e-02, -1.701821737645238e-02, -5.859663549756466e-02, -5.735031582855032e-02, -2.433648372205916e-02, -2.217588045264967e-02, -2.433648372205917e-02, -2.217588045264967e-02, -5.340581179423598e-03, -5.068042664081127e-03, -7.404271625153550e-04, -7.382954736025375e-04, -1.513996383180355e-03, -1.396773793331012e-03, -2.268459559752971e-02, -2.205172440630947e-02, -2.552007634584141e-03, -1.769144082081008e-03, -2.552007634584142e-03, -1.769144082081009e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_am05_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.040534596845582e-10, 0.000000000000000e+00, 1.040543796758561e-10, 1.040549667732646e-10, 0.000000000000000e+00, 1.040554873685159e-10, 1.040553482558045e-10, 0.000000000000000e+00, 1.040571901641993e-10, 1.040382528404728e-10, 0.000000000000000e+00, 1.040355831768943e-10, 1.040544812532147e-10, 0.000000000000000e+00, 1.040409893565385e-10, 1.040544812532147e-10, 0.000000000000000e+00, 1.040409893565385e-10, 4.486989712805121e-07, 0.000000000000000e+00, 4.493351178856303e-07, 4.487679776735903e-07, 0.000000000000000e+00, 4.494471667350539e-07, 4.508831628427974e-07, 0.000000000000000e+00, 4.514243466060246e-07, 4.468850635129007e-07, 0.000000000000000e+00, 4.475167759792279e-07, 4.488387137759029e-07, 0.000000000000000e+00, 4.493089200304752e-07, 4.488387137759029e-07, 0.000000000000000e+00, 4.493089200304752e-07, 5.418651976671136e-04, 0.000000000000000e+00, 5.697501251033060e-04, 5.330278226314388e-04, 0.000000000000000e+00, 5.667544308832064e-04, 4.301603766818835e-04, 0.000000000000000e+00, 3.972506959714530e-04, 4.004264159762833e-04, 0.000000000000000e+00, 4.113499873561129e-04, 6.027421021807699e-04, 0.000000000000000e+00, 2.623505130620393e-04, 6.027421021807699e-04, 0.000000000000000e+00, 2.623505130620393e-04, 2.930168176833210e-02, 0.000000000000000e+00, 2.921650416838386e-02, 3.098086029164553e-02, 0.000000000000000e+00, 3.072860716690584e-02, 2.965955982755090e-04, 0.000000000000000e+00, 3.303563392453947e-04, 2.491589130493074e-02, 0.000000000000000e+00, 2.409285017666525e-02, 1.817695675034429e-02, 0.000000000000000e+00, 2.867451683314776e-02, 1.817695675034428e-02, 0.000000000000000e+00, 2.867451683314780e-02, 7.878525995714809e-02, 0.000000000000000e+00, 7.342064777162485e-02, 8.268147950368208e-02, 0.000000000000000e+00, 7.681659892715859e-02, 3.269119671539920e-02, 0.000000000000000e+00, 3.135377012388411e-02, 7.483305882272570e-02, 0.000000000000000e+00, 7.403250708654513e-02, 5.717807672863004e-02, 0.000000000000000e+00, 2.547112254049164e-01, 5.717807672863013e-02, 0.000000000000000e+00, 2.547112254049169e-01, 1.159185355969043e-07, 0.000000000000000e+00, 1.160296919344042e-07, 1.165005424372901e-07, 0.000000000000000e+00, 1.165928683001826e-07, 1.159471534293700e-07, 0.000000000000000e+00, 1.160464937447749e-07, 1.164466600525370e-07, 0.000000000000000e+00, 1.165579750953737e-07, 1.162190300693888e-07, 0.000000000000000e+00, 1.163126835630199e-07, 1.162190300693888e-07, 0.000000000000000e+00, 1.163126835630199e-07, 2.219479350125583e-06, 0.000000000000000e+00, 2.219835890165416e-06, 2.218101773845889e-06, 0.000000000000000e+00, 2.219332680705278e-06, 2.088685938480749e-06, 0.000000000000000e+00, 2.126477700955805e-06, 2.088885328947999e-06, 0.000000000000000e+00, 2.124838618161470e-06, 2.317028402062029e-06, 0.000000000000000e+00, 2.235280080881316e-06, 2.317028402062029e-06, 0.000000000000000e+00, 2.235280080881316e-06, 2.180857798890054e-03, 0.000000000000000e+00, 2.216035771826422e-03, 3.542288023312217e-03, 0.000000000000000e+00, 3.576222467126807e-03, 2.880688565050697e-03, 0.000000000000000e+00, 2.640772925164323e-03, 5.935266063731942e-03, 0.000000000000000e+00, 4.967935331598683e-03, 2.043912399998481e-03, 0.000000000000000e+00, 2.531911837617785e-03, 2.043912399998483e-03, 0.000000000000000e+00, 2.531911837617785e-03, 2.121496916740449e-02, 0.000000000000000e+00, 2.160254008908916e-02, 1.218288033040354e-02, 0.000000000000000e+00, 1.209441455020329e-02, 2.392854749218215e-02, 0.000000000000000e+00, 2.277840562454571e-02, 2.101476917504867e-05, 0.000000000000000e+00, 2.107503198346758e-05, 2.584474919496510e-02, 0.000000000000000e+00, 3.352881959859666e-02, 2.584474919496510e-02, 0.000000000000000e+00, 3.352881959859666e-02, 1.112044883163406e-01, 0.000000000000000e+00, 9.261712332748750e-02, 9.216486532672907e-02, 0.000000000000000e+00, 8.382430500374355e-02, 5.525308820056561e-01, 0.000000000000000e+00, 5.805082059128035e-01, 3.517152133789411e-02, 0.000000000000000e+00, 3.364212727771061e-02, 2.994572068587972e-01, 0.000000000000000e+00, 2.143353714958053e-01, 2.994572068587965e-01, 0.000000000000000e+00, 2.143353714958048e-01, 6.826942892691118e-03, 0.000000000000000e+00, 6.727925523633670e-03, 5.380221662603790e-03, 0.000000000000000e+00, 5.316771320775582e-03, 5.828855112605466e-03, 0.000000000000000e+00, 5.757961725446920e-03, 6.250124213991208e-03, 0.000000000000000e+00, 6.162847316459401e-03, 6.034023655909910e-03, 0.000000000000000e+00, 5.955081840311817e-03, 6.034023655909910e-03, 0.000000000000000e+00, 5.955081840311817e-03, 8.442109583173618e-03, 0.000000000000000e+00, 8.309683741780025e-03, 2.419593961477563e-03, 0.000000000000000e+00, 2.409024973684446e-03, 3.226906664183996e-03, 0.000000000000000e+00, 3.219005760744333e-03, 4.569922815866176e-03, 0.000000000000000e+00, 4.521567855335155e-03, 3.831477110672206e-03, 0.000000000000000e+00, 3.789644532396460e-03, 3.831477110672206e-03, 0.000000000000000e+00, 3.789644532396460e-03, 2.736392107503067e-03, 0.000000000000000e+00, 2.763320463848298e-03, 8.779951308246938e-03, 0.000000000000000e+00, 8.744346700723429e-03, 8.215301821088407e-03, 0.000000000000000e+00, 8.287616757350968e-03, 9.793848549072596e-03, 0.000000000000000e+00, 9.674377532660211e-03, 9.039425838615596e-03, 0.000000000000000e+00, 9.174700797371141e-03, 9.039425838615604e-03, 0.000000000000000e+00, 9.174700797371149e-03, 3.415748222219526e-03, 0.000000000000000e+00, 3.410924257341180e-03, 2.949224186168600e-02, 0.000000000000000e+00, 2.934958502955494e-02, 2.780120103086322e-02, 0.000000000000000e+00, 2.770153742283184e-02, 1.791895839979490e-02, 0.000000000000000e+00, 1.749013356264889e-02, 3.611725104796666e-02, 0.000000000000000e+00, 4.257133779730561e-02, 3.611725104796669e-02, 0.000000000000000e+00, 4.257133779730561e-02, 5.837669101665033e-02, 0.000000000000000e+00, 5.753032108631774e-02, 3.687764814071223e-01, 0.000000000000000e+00, 6.517665101143065e-01, 2.209374030923847e-01, 0.000000000000000e+00, 2.211557936968764e-01, 4.022338679987666e-02, 0.000000000000000e+00, 3.899305190287385e-02, 6.154569153439046e-01, 0.000000000000000e+00, 2.291213916244779e-01, 6.154569153439066e-01, 0.000000000000000e+00, 2.291213916244789e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05