
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vbh_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vbh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.722760042779017e-01, -1.722760744758199e-01, -1.722764082236336e-01, -1.722753633492232e-01, -1.722759077486037e-01, -1.722759077486037e-01, -1.266824212000454e-01, -1.266824752035873e-01, -1.266843366079262e-01, -1.266858516815564e-01, -1.266830753644471e-01, -1.266830753644471e-01, -8.593836019569505e-02, -8.591106114968682e-02, -8.525547233751242e-02, -8.545885348104833e-02, -8.516245640029030e-02, -8.516245640029030e-02, -5.417750796069749e-02, -5.449662797086956e-02, -8.961453640892189e-02, -4.822746190022511e-02, -4.359553766411746e-02, -4.359553766411747e-02, -6.080695156297090e-03, -6.361866253412479e-03, -2.365198098525123e-02, -3.695861725577592e-03, -4.784003136354923e-03, -4.784003136354923e-03, -1.367943239177905e-01, -1.367989277599010e-01, -1.367945530168340e-01, -1.367986175882452e-01, -1.367966482929786e-01, -1.367966482929786e-01, -1.123326732684421e-01, -1.125125434445235e-01, -1.122002873662068e-01, -1.123601263334400e-01, -1.124940097848096e-01, -1.124940097848096e-01, -8.269283577123472e-02, -8.470024603994751e-02, -8.074589475447909e-02, -8.169321814702568e-02, -8.293060344903914e-02, -8.293060344903912e-02, -4.067992968882380e-02, -5.425569869151752e-02, -3.908646337953629e-02, -1.118522013532807e-01, -4.385966124822663e-02, -4.385966124822663e-02, -2.901864612851203e-03, -3.618963219995307e-03, -2.818150640152820e-03, -3.168016509664978e-02, -3.437420220811910e-03, -3.437420220811911e-03, -8.280958209414838e-02, -8.267037806311009e-02, -8.271944412751808e-02, -8.275980171203387e-02, -8.273960921771947e-02, -8.273960921771947e-02, -8.214520463773099e-02, -7.846693958798656e-02, -7.957452640808324e-02, -8.062743007406167e-02, -8.009530794727748e-02, -8.009530794727748e-02, -8.584745737981088e-02, -5.949972530260536e-02, -6.368943627542434e-02, -7.018341268655350e-02, -6.691011227159499e-02, -6.691011227159499e-02, -7.648404386659891e-02, -2.301246551000922e-02, -2.790625742174882e-02, -6.931254062528869e-02, -3.608926647013828e-02, -3.608926647013828e-02, -8.190142077706773e-03, -1.022569357986819e-03, -2.100598056258705e-03, -3.494584284686000e-02, -3.185689475233249e-03, -3.185689475233099e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vbh_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vbh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.806660675726696e-01, -1.806665552432564e-01, -1.806659704636387e-01, -1.806667928037928e-01, -1.806671700906254e-01, -1.806662609296579e-01, -1.806642278822817e-01, -1.806671126145376e-01, -1.806638014990521e-01, -1.806686282728286e-01, -1.806638014990521e-01, -1.806686282728286e-01, -1.350215508211836e-01, -1.350251382497920e-01, -1.350213504412880e-01, -1.350254469108869e-01, -1.350279709771987e-01, -1.350225579610965e-01, -1.350286281085461e-01, -1.350249379790098e-01, -1.350732074100072e-01, -1.349748460215385e-01, -1.350732074100072e-01, -1.349748460215385e-01, -9.436198229948289e-02, -9.372755043206440e-02, -9.440749488009167e-02, -9.362860747211754e-02, -9.280897459933032e-02, -9.390586319547924e-02, -9.373061697673228e-02, -9.338615426928186e-02, -8.853942372266503e-02, -9.893545091362536e-02, -8.853942372266503e-02, -9.893545091362536e-02, -6.322396332452322e-02, -6.004865695040529e-02, -6.381621643851317e-02, -6.017085356674104e-02, -1.006949439813901e-01, -9.508570711843657e-02, -5.582076343247526e-02, -5.486991272321768e-02, -4.364535707765812e-02, -8.545904105416180e-02, -4.364535707765814e-02, -8.545904105416179e-02, -7.786195916277774e-03, -7.937990275443452e-03, -8.130337841792179e-03, -8.296502285284713e-03, -2.924592017849746e-02, -2.824246083007265e-02, -4.858440682826346e-03, -4.814317431571797e-03, -6.384686817238247e-03, -4.992416460868013e-03, -6.384686817238247e-03, -4.992416460868013e-03, -1.451201923559675e-01, -1.451893234475506e-01, -1.451238907319582e-01, -1.451948484760812e-01, -1.451197789134164e-01, -1.451901967065100e-01, -1.451243779028692e-01, -1.451937388664767e-01, -1.451218865936919e-01, -1.451922861368580e-01, -1.451218865936919e-01, -1.451922861368580e-01, -1.206227061227141e-01, -1.206341581429578e-01, -1.207781496188725e-01, -1.208399559371838e-01, -1.206915107057571e-01, -1.203004555091013e-01, -1.208605731458693e-01, -1.204524759595432e-01, -1.202953376343319e-01, -1.212917955227755e-01, -1.202953376343319e-01, -1.212917955227755e-01, -9.055611973811933e-02, -9.095988261130444e-02, -9.282585033820667e-02, -9.275445970588483e-02, -9.158342798474048e-02, -8.625471966745560e-02, -9.230724930464956e-02, -8.740685205659035e-02, -8.736595777281335e-02, -9.519419067435471e-02, -8.736595777281332e-02, -9.519419067435467e-02, -4.760213815479832e-02, -4.710407749351197e-02, -6.184327063701151e-02, -6.142846012907200e-02, -4.702032317849822e-02, -4.445177745905689e-02, -1.200953462454614e-01, -1.201966397354809e-01, -5.258141094141262e-02, -4.914601305201802e-02, -5.258141094141262e-02, -4.914601305201802e-02, -3.763553326093503e-03, -3.855663429882926e-03, -4.718006205899211e-03, -4.757214585973567e-03, -3.630174972701184e-03, -3.765387714330704e-03, -3.773343159137608e-02, -3.752774896615759e-02, -3.925471408523343e-03, -4.707206832030976e-03, -3.925471408523343e-03, -4.707206832030976e-03, -9.117150586952751e-02, -9.058275850417771e-02, -9.103278465747418e-02, -9.043946192383208e-02, -9.108263113449197e-02, -9.048904274314953e-02, -9.112097572770049e-02, -9.053241929203876e-02, -9.110175987803834e-02, -9.051074630405891e-02, -9.110175987803834e-02, -9.051074630405891e-02, -9.045255246722751e-02, -8.995438886545278e-02, -8.677918760801261e-02, -8.616599927445620e-02, -8.791729378237452e-02, -8.727706299058559e-02, -8.892305891373119e-02, -8.840641395773469e-02, -8.839548813616292e-02, -8.785462937984535e-02, -8.839548813616292e-02, -8.785462937984535e-02, -9.408170360895364e-02, -9.382095715889102e-02, -6.734575146860809e-02, -6.679311326125306e-02, -7.191760041179374e-02, -7.086214310780048e-02, -7.837268403000601e-02, -7.770270638635560e-02, -7.466684276960910e-02, -7.470690294226764e-02, -7.466684276960910e-02, -7.470690294226764e-02, -8.493021904545162e-02, -8.399035653285156e-02, -2.804675474881068e-02, -2.792890579823605e-02, -3.392045103431066e-02, -3.305823355148902e-02, -7.805737372565175e-02, -7.627041795614123e-02, -4.366707181944460e-02, -4.134574533246700e-02, -4.366707181944460e-02, -4.134574533246700e-02, -1.047330226014096e-02, -1.052896713352422e-02, -1.354841057373707e-03, -1.357492221426607e-03, -2.703376297783649e-03, -2.826155386583271e-03, -4.151936576214217e-02, -4.086671147444677e-02, -3.688317567729631e-03, -4.373416703122352e-03, -3.688317567729384e-03, -4.373416703122399e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05