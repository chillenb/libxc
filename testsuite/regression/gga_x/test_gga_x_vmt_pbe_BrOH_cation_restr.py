
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_vmt_pbe_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.096072789845982e+01, -2.096075453227179e+01, -2.096094203277933e+01, -2.096054435550536e+01, -2.096074290252260e+01, -2.096074290252260e+01, -3.477980229134342e+00, -3.477940574550155e+00, -3.477109342576550e+00, -3.479215356150030e+00, -3.478016524017268e+00, -3.478016524017268e+00, -6.996168219512400e-01, -6.996785218310022e-01, -7.033396682281636e-01, -7.080460709421648e-01, -7.061427500297497e-01, -7.061427500297497e-01, -2.201780651769936e-01, -2.209556375609178e-01, -8.071109189752046e-01, -1.895974442299181e-01, -2.012260559392231e-01, -2.012260559392231e-01, -5.598781768267900e-03, -5.898483972720346e-03, -5.611342269591524e-02, -3.231372209763798e-03, -4.060475926619071e-03, -4.060475926619071e-03, -5.036943344436875e+00, -5.036295896148906e+00, -5.036924444706036e+00, -5.036352702797654e+00, -5.036609068433150e+00, -5.036609068433150e+00, -2.126075645351139e+00, -2.135541410531371e+00, -2.128450186308534e+00, -2.136790837902993e+00, -2.130167322924215e+00, -2.130167322924215e+00, -5.813021359057288e-01, -6.028709928930405e-01, -5.423262980029089e-01, -5.374166841651290e-01, -5.869396228564637e-01, -5.869396228564637e-01, -1.467719883079087e-01, -2.389922126636724e-01, -1.371735755870876e-01, -1.813749718788696e+00, -1.627394708472993e-01, -1.627394708472993e-01, -2.493218703619550e-03, -3.158952218556460e-03, -2.415487312386814e-03, -9.500741745021583e-02, -2.909913205573093e-03, -2.909913205573093e-03, -5.507685109649985e-01, -5.539516135206416e-01, -5.528446914070662e-01, -5.519139364972870e-01, -5.523798124628643e-01, -5.523798124628643e-01, -5.339849933753127e-01, -5.121062934653405e-01, -5.178306992847402e-01, -5.234629743580989e-01, -5.203440980239163e-01, -5.203440980239163e-01, -6.331731436361384e-01, -2.824035310057715e-01, -3.152139376050168e-01, -3.672747290677871e-01, -3.386687414072663e-01, -3.386687414072663e-01, -4.726843325680096e-01, -5.293346083282435e-02, -7.579791857591762e-02, -3.429457797983088e-01, -1.181195712496624e-01, -1.181195712496624e-01, -8.027045986457013e-03, -8.444081655323325e-04, -1.772450099569052e-03, -1.118533260841607e-01, -2.692579680327262e-03, -2.692579680327260e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_vmt_pbe_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.508853742791421e+01, -2.508863993377948e+01, -2.508906504950954e+01, -2.508754060450615e+01, -2.508834373045204e+01, -2.508834373045204e+01, -4.021367994302789e+00, -4.021419704564261e+00, -4.022714401093802e+00, -4.021227989524048e+00, -4.021526933326943e+00, -4.021526933326943e+00, -7.530082058114376e-01, -7.516309896326145e-01, -7.193817297046295e-01, -7.258697750434288e-01, -7.249074535628166e-01, -7.249074535628166e-01, -1.953006283696532e-01, -1.965384746300760e-01, -8.840586822110886e-01, -1.769389793900535e-01, -1.801698875403561e-01, -1.801698875403561e-01, -7.536147379171178e-03, -8.025403159121239e-03, -8.930046249837227e-02, -4.308496298194733e-03, -5.414293530775029e-03, -5.414293530775029e-03, -6.192558120642033e+00, -6.195431620384972e+00, -6.192686438565977e+00, -6.195223224558681e+00, -6.194016686416496e+00, -6.194016686416496e+00, -2.151931238809251e+00, -2.170001326220057e+00, -2.135307853310853e+00, -2.151061896653437e+00, -2.170208425691072e+00, -2.170208425691072e+00, -6.836552938673908e-01, -7.741391574378637e-01, -6.283935702933378e-01, -6.810292332988740e-01, -6.983580164685029e-01, -6.983580164685029e-01, -1.631209196201756e-01, -2.151920412465157e-01, -1.565149940734060e-01, -2.333832675821146e+00, -1.622231902575992e-01, -1.622231902575992e-01, -3.324291604826534e-03, -4.211936331022699e-03, -3.220674218117307e-03, -1.234481468525390e-01, -3.879909815658395e-03, -3.879909815658395e-03, -7.247524245858927e-01, -7.125653807438073e-01, -7.167915201819451e-01, -7.203336494741259e-01, -7.185564195069760e-01, -7.185564195069760e-01, -7.075300022453702e-01, -5.537651355276950e-01, -5.943085422755637e-01, -6.380568546598210e-01, -6.155454188830336e-01, -6.155454188830336e-01, -8.101844147557447e-01, -2.500612796388250e-01, -2.872954638873876e-01, -3.873730134278031e-01, -3.293154323697768e-01, -3.293154323697767e-01, -5.090021956485157e-01, -8.664143774335425e-02, -1.094508269199759e-01, -3.809758390988374e-01, -1.369040469625415e-01, -1.369040469625416e-01, -1.212899506294494e-02, -1.125877554043110e-03, -2.363266799425403e-03, -1.332484338757380e-01, -3.590119686358471e-03, -3.590119686358468e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_vmt_pbe_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.238786905677995e-09, -7.238731202687457e-09, -7.238418463458360e-09, -7.239248558902332e-09, -7.238822852959676e-09, -7.238822852959676e-09, -9.857549459062129e-06, -9.857782086346945e-06, -9.862191131264093e-06, -9.847040748411211e-06, -9.856877896149114e-06, -9.856877896149114e-06, -6.413564288287385e-03, -6.421833262776795e-03, -6.551024780165970e-03, -6.366708220160730e-03, -6.428529327370014e-03, -6.428529327370014e-03, -7.287512763495804e-01, -7.227411553764073e-01, -3.566480746554184e-03, -1.029143447166926e+00, -9.570995243161889e-01, -9.570995243161888e-01, 1.050693375029917e+01, 2.094309378255046e+01, 8.683982432196956e+00, 1.352273335385998e-05, 1.351916165756141e-01, 1.351916165756162e-01, -2.117076819782368e-06, -2.116981970439086e-06, -2.117059718183424e-06, -2.116976188714639e-06, -2.117035900725887e-06, -2.117035900725887e-06, -7.911928872732097e-05, -7.748430504257481e-05, -7.931716825715337e-05, -7.787651277135412e-05, -7.810597477116796e-05, -7.810597477116796e-05, -1.243736288933562e-02, -9.895977918439498e-03, -1.664161968062267e-02, -1.587242324439177e-02, -1.183889876888659e-02, -1.183889876888659e-02, -1.097368997654663e+00, -4.696804962011825e-01, -1.150297644408803e+00, -1.205529985701134e-04, -1.421314525287043e+00, -1.421314525287043e+00, 8.538567696882954e-10, 3.360684719590578e-05, 1.163778230753430e-01, -5.133969171523494e-01, 4.740008699557544e-02, 4.740008699557451e-02, -1.387176966256948e-02, -1.385910455532141e-02, -1.386356881259653e-02, -1.386767122272979e-02, -1.386568269064629e-02, -1.386568269064629e-02, -1.559373059723927e-02, -2.225161404030391e-02, -2.019491349424865e-02, -1.829659540791726e-02, -1.926830887900689e-02, -1.926830887900689e-02, -8.161085229902949e-03, -2.667004713598836e-01, -1.752263703758533e-01, -8.590799287512560e-02, -1.266771955021985e-01, -1.266771955021985e-01, -3.076639682188178e-02, 1.044559242826537e+01, 2.205905873250636e+00, -1.080712889991470e-01, -1.799855844582743e+00, -1.799855844582739e+00, 6.737744797180513e+01, 2.219140290780102e-54, 1.816769196440856e-14, -1.651577325785452e+00, 3.539673409219476e-02, 3.539673409219458e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05