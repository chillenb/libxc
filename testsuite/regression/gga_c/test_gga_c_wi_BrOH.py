
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_wi_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.456315344165781e-01, -2.456325087057841e-01, -2.456355869014665e-01, -2.456196570907190e-01, -2.456320588261873e-01, -2.456320588261873e-01, -1.459243075054929e-01, -1.459316137303558e-01, -1.461348551734950e-01, -1.457981084261762e-01, -1.459283409635763e-01, -1.459283409635763e-01, -3.541498699046539e-02, -3.502401873552722e-02, -2.450618472142166e-02, -2.519137459176709e-02, -3.527407748681322e-02, -3.527407748681322e-02, -1.366991009420112e-03, -1.787467498458632e-03, -5.206733949348157e-02, 7.247440382185561e-04, -1.493067520569033e-03, -1.493067520569033e-03, -1.742566868917017e-08, -2.153996421492025e-08, 1.206482027818508e-05, -4.689741419568785e-10, -2.124421507434033e-08, -2.124421507434033e-08, -1.927567888997194e-01, -1.929474565709117e-01, -1.927759309120225e-01, -1.929241966933850e-01, -1.928551705623072e-01, -1.928551705623072e-01, -5.972891350497286e-02, -6.156992803641878e-02, -5.743938784933879e-02, -5.887659790604791e-02, -6.396992672450552e-02, -6.396992672450552e-02, -4.817229290167215e-02, -6.280455675572241e-02, -4.509555907338694e-02, -5.652927492515000e-02, -5.172948534938848e-02, -5.172948534938848e-02, 3.840480999814574e-04, 6.747343660836807e-04, 3.972286115547523e-04, -1.337676547788014e-01, 5.930320357112465e-04, 5.930320357112465e-04, -4.200492150825897e-10, -7.678162839672178e-10, -5.263850810229821e-10, 8.837323487867120e-05, -7.106451029197624e-10, -7.106451029197620e-10, -6.233889342569052e-02, -6.041033653977361e-02, -6.113587962758250e-02, -6.167169456553653e-02, -6.140666132720381e-02, -6.140666132720381e-02, -6.097065125240579e-02, -3.282234688998981e-02, -4.241018703436535e-02, -5.072612543217104e-02, -4.669922795961013e-02, -4.669922795961013e-02, -6.487308361342893e-02, -5.825259956532177e-04, -4.544817687944453e-03, -2.047287252574189e-02, -1.119053566074659e-02, -1.119053566074659e-02, -2.845183697919493e-02, 4.310267357130962e-06, 4.306896042927996e-05, -2.494221521234701e-02, 2.598550936775181e-04, 2.598550936775184e-04, -2.776511377446328e-08, -3.620225995354840e-12, -5.703391771129293e-11, 2.506388370297988e-04, -4.759020384593691e-10, -4.759020384593652e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_wi_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.483890587470847e-01, -3.483874996263785e-01, -3.483829027508365e-01, -3.484083881159699e-01, -3.483882168022779e-01, -3.483882168022779e-01, -2.838146245052616e-01, -2.838025991188123e-01, -2.834757283762730e-01, -2.840576666510219e-01, -2.838088161816013e-01, -2.838088161816013e-01, -1.278307189407281e-01, -1.276662492470011e-01, -1.172566251258253e-01, -1.187226640689489e-01, -1.277727469126567e-01, -1.277727469126567e-01, -1.716494074088575e-02, -1.945181776889477e-02, -1.419608707061787e-01, 4.928862205789107e-04, -1.786494304028169e-02, -1.786494304028169e-02, -1.045529469288390e-07, -1.292381782968862e-07, 1.189577284767280e-04, -2.813843257523380e-09, -1.274637130354504e-07, -1.274637130354504e-07, -2.752952178517964e-01, -2.749588166504782e-01, -2.752616159171903e-01, -2.750000227867330e-01, -2.751213936870797e-01, -2.751213936870797e-01, -2.554423114867997e-01, -2.573526083520746e-01, -2.535271103943225e-01, -2.551992143128187e-01, -2.586428854285719e-01, -2.586428854285719e-01, -1.024589783373492e-01, -8.804578168780768e-02, -9.998009968177965e-02, -8.563475917615243e-02, -1.020860685032971e-01, -1.020860685032971e-01, 1.407747298402505e-03, -3.575967497675530e-03, 1.401301933750594e-03, -1.753388473560921e-01, 1.412251086248553e-03, 1.412251086248553e-03, -2.520293964842900e-09, -4.606893826829984e-09, -3.158308056484939e-09, 5.148253593795251e-04, -4.263866973362863e-09, -4.263866973362862e-09, -8.087805993807208e-02, -8.364470028416582e-02, -8.254934585453813e-02, -8.177651746481632e-02, -8.215404922168867e-02, -8.215404922168867e-02, -7.850222671036156e-02, -1.016893964515614e-01, -9.769045772615279e-02, -8.928330503174070e-02, -9.368960863928547e-02, -9.368960863928544e-02, -9.180882360346895e-02, -1.508042580285880e-02, -3.699181237178699e-02, -7.544373997937776e-02, -5.951880421314394e-02, -5.951880421314393e-02, -9.394875345087786e-02, 5.819589531872815e-05, 3.073735494995033e-04, -7.059084370889157e-02, 1.073487924548964e-03, 1.073487924548964e-03, -1.665880022537743e-07, -2.172135565788251e-11, -3.422034651851948e-10, 1.028990058784458e-03, -2.855410308418546e-09, -2.855410308418523e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_wi_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wi", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.377763076197060e-10, 2.377731930329753e-10, 2.377575046758640e-10, 2.378085445732804e-10, 2.377746800848974e-10, 2.377746800848974e-10, 1.861737952750388e-06, 1.861706526547992e-06, 1.860470667634548e-06, 1.860708575817310e-06, 1.861683921958607e-06, 1.861683921958607e-06, 2.805335281393068e-03, 2.798111965298491e-03, 2.434524940759613e-03, 2.395359716227874e-03, 2.802838347845723e-03, 2.802838347845723e-03, 1.195028491846984e-01, 1.329265172883630e-01, 1.471498823318592e-03, 7.106259864978736e-03, 1.237313809141226e-01, 1.237313809141226e-01, 3.630698602474313e-03, 3.912631542281634e-03, -4.099101338592082e-02, 7.414393981135150e-04, 4.058644919812666e-03, 4.058644919812666e-03, 2.501924893990376e-07, 2.497433750737457e-07, 2.501464378364567e-07, 2.497972361872339e-07, 2.499624923225747e-07, 2.499624923225747e-07, 2.101138101851248e-05, 2.072442298515828e-05, 2.073544238057727e-05, 2.053208145528605e-05, 2.112271076394679e-05, 2.112271076394679e-05, 5.990108337128570e-03, 3.282500605097751e-03, 7.507532594946270e-03, 5.368554822591897e-03, 5.043937144182797e-03, 5.043937144182797e-03, -2.864098090185366e-02, 1.889128536277717e-02, -3.140939523983294e-02, 2.429992757607407e-05, -1.502372732626892e-02, -1.502372732626892e-02, 7.618592510289364e-04, 9.342584501052470e-04, 2.573415436452918e-03, -6.160466253072763e-02, 1.389276123447519e-03, 1.389276123447517e-03, 3.341800052563191e-03, 4.074189182175624e-03, 3.835282870743139e-03, 3.634896801848474e-03, 3.737051543395193e-03, 3.737051543395193e-03, 3.676537129328590e-03, 1.023703429879505e-02, 9.180450263688134e-03, 7.394484576439409e-03, 8.376426589361540e-03, 8.376426589361536e-03, 2.729106926405595e-03, 2.848524293199722e-02, 4.010818599243880e-02, 4.186331267286329e-02, 4.680740081523050e-02, 4.680740081523053e-02, 1.532123662135481e-02, -2.384740720256066e-02, -4.785446373835042e-02, 6.137516515772040e-02, -5.461441812408144e-02, -5.461441812408146e-02, 3.118134722104458e-03, 2.811621072224054e-04, 5.997766995696538e-04, -6.674000852244898e-02, 1.831309475400520e-03, 1.831309475400511e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05