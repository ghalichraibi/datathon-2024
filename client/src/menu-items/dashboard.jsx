// assets
import { DashboardOutlined, UploadOutlined } from '@ant-design/icons';

// icons
const icons = {
  DashboardOutlined,
  UploadOutlined
};

// ==============================|| MENU ITEMS - DASHBOARD ||============================== //

const dashboard = {
  id: 'group-dashboard',
  title: 'Navigation',
  type: 'group',
  children: [
    { 
      id: 'upload',
      title: 'My Reports',
      type: 'item',
      url: '/upload',
      icon: icons.UploadOutlined,
      breadcrumbs: false
    },
    {
      id: 'dashboard',
      title: 'Dashboard',
      type: 'item',
      url: '/dashboard',
      icon: icons.DashboardOutlined,
      breadcrumbs: false
    }
  ]
};

export default dashboard;
